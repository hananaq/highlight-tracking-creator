# app.py —  Highlight & Tracking Creator 
# by Eng.Hanan Ak (https://www.linkedin.com/in/hananabukwaider/)


import os, json, tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st

# Optional deps
try:
    import cv2
except Exception:
    cv2 = None
try:
    import numpy as np
except Exception:
    np = None
try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    ssim = None
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

DEFAULT_MODEL = "yolo11s.pt"
FRAMES_PER_KEYFRAME = 2          # short hold = fast-forward feel
FIXED_MIN_FRAME_GAP = 15         # debounce frames between kept keyframes
FIXED_DOWNSCALE = 2              # speed up analysis without losing structure

# --- COCO-80 class names ---
CLASS_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]
TRACK_OPTIONS = {name: i for i, name in enumerate(CLASS_NAMES)}

# ---------- Utils ----------
def ensure_deps():
    missing = []
    if cv2 is None: missing.append("opencv-python")
    if np is None: missing.append("numpy")
    if ssim is None: missing.append("scikit-image")
    if YOLO is None: missing.append("ultralytics")
    return missing

def open_video_capture(path: str):
    if cv2 is None:
        raise RuntimeError("OpenCV not installed.")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Could not open video. Re-encode to H.264 MP4.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cv2.CAP_PROP_FRAME_COUNT and cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, float(fps), total, w, h

def robust_mp4_writer(base_path: Path, fps: float, size: Tuple[int, int]):
    w, h = size
    trials = [("avc1",".mp4"),("H264",".mp4"),("mp4v",".mp4")]
    for fourcc_name, ext in trials:
        out_path = base_path.with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(str(out_path), fourcc, max(1.0, fps), (w, h))
        if writer.isOpened():
            return writer, out_path
    raise RuntimeError("Failed to open MP4 writer. Ensure ffmpeg/x264 is available.")

def validate_nonempty_video(path: Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release(); return False
    total = int(cv2.CAP_PROP_FRAME_COUNT and cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, _ = cap.read()
    cap.release()
    return ok and total > 0

def _read_bytes(p: Path) -> bytes:
    with open(p, "rb") as f:
        return f.read()

# ---- Label drawing with background (visible on dark frames) ----
def draw_label_bg(img, text, x, y, scale=0.6, fg=(255,255,255), bg=(0,0,0)):
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 2)
    pad = 3
    cv2.rectangle(img, (x, y - th - pad), (x + tw + pad*2, y + baseline), bg, thickness=-1)
    cv2.putText(img, text, (x + pad, y - pad), font, scale, fg, 2, cv2.LINE_AA)

# ---- aHash / SSIM / Flow helpers ----
def to_gray_small(img: np.ndarray, size: int = 8) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return g

def ahash_64(gray8x8: np.ndarray) -> int:
    avg = gray8x8.mean()
    bits = (gray8x8 >= avg).astype(np.uint8)
    val = 0
    for b in bits.flatten():
        val = (val << 1) | int(b)
    return int(val)

def hamming_dist64(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))

def compute_flow_mag(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    return float(np.mean(mag))

# ---- Single sensitivity control -> thresholds ----
def thresholds_from_sensitivity(sens: float):
    sens = max(0.1, min(1.0, float(sens)))   # 0.1..1.0
    ahash_min, ahash_max = 6, 18
    ahash_thresh = int(round(ahash_max - (ahash_max - ahash_min) * sens))
    ssim_min, ssim_max = 0.08, 0.25
    ssim_thresh = ssim_max - (ssim_max - ssim_min) * sens
    flow_min, flow_max = 0.4, 1.2
    flow_thresh = flow_max - (flow_max - flow_min) * sens
    return ahash_thresh, ssim_thresh, flow_thresh

# ---- Keyframe selection ----
def select_keyframes_by_change(video_path: Path, out_dir: Path, sensitivity: float) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap, fps, total, w, h = open_video_capture(str(video_path))
    ahash_thresh, ssim_thresh, flow_thresh = thresholds_from_sensitivity(sensitivity)

    last_keep_idx = -10**9
    last_gray_small = None
    last_hash = None
    kept_paths: List[Path] = []
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_proc = cv2.resize(frame, (w // FIXED_DOWNSCALE, h // FIXED_DOWNSCALE),
                                interpolation=cv2.INTER_AREA) if FIXED_DOWNSCALE > 1 else frame
        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
        gray_ssim = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        hash8 = to_gray_small(frame_proc, size=8)
        cur_hash = ahash_64(hash8)

        keep = False
        if idx - last_keep_idx >= FIXED_MIN_FRAME_GAP:
            if last_gray_small is None:
                keep = True
            else:
                hd = hamming_dist64(cur_hash, last_hash) if last_hash is not None else 64
                try:
                    from skimage.metrics import structural_similarity as ssim_metric
                    s = ssim_metric(gray_ssim, last_gray_small, data_range=gray_ssim.max() - gray_ssim.min())
                    dissim = 1.0 - float(s)
                except Exception:
                    dissim = 0.0
                flow_mag = compute_flow_mag(last_gray_small, gray_ssim) if last_gray_small is not None else 10.0
                if hd >= ahash_thresh or dissim >= ssim_thresh or flow_mag >= flow_thresh:
                    keep = True

        if keep:
            out_path = out_dir / f"{idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)  
            kept_paths.append(out_path)
            last_keep_idx = idx
            last_gray_small = gray_ssim
            last_hash = cur_hash

        idx += 1

    cap.release()
    return kept_paths

def build_highlight_from_keyframes(video_path: Path, keyframes: List[Path], out_base: Path) -> Path:
    if not keyframes:
        raise RuntimeError("No keyframes to build highlight.")
    cap, fps, total, w, h = open_video_capture(str(video_path))
    cap.release()
    writer, out_path = robust_mp4_writer(out_base, fps, (w, h))
    frames_written = 0

    for kf in keyframes:
        img = cv2.imread(str(kf))
        if img is None:
            continue
        if (img.shape[1], img.shape[0]) != (w, h):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        for _ in range(FRAMES_PER_KEYFRAME):
            writer.write(img)
            frames_written += 1

    writer.release()
    if frames_written == 0 or not validate_nonempty_video(out_path):
        raise RuntimeError("Highlight MP4 empty/invalid.")
    return out_path

# ---- Tracking with JSON export of centroids ----
def run_tracking_to_mp4_with_json(video_path: Path, classes: List[int], conf: float, out_base: Path):
    if YOLO is None:
        raise RuntimeError("Ultralytics not installed.")
    model = YOLO(DEFAULT_MODEL)

    cap, fps, total, w, h = open_video_capture(str(video_path))
    cap.release()

    writer, out_path = robust_mp4_writer(out_base, max(1.0, float(fps)), (w, h))
    trajectories = {
        "fps": fps,
        "width": w,
        "height": h,
        "classes_tracked": classes,
        "records": []  # flat list: {id, frame, cx, cy}
    }

    stream = model.track(
        source=str(video_path),
        conf=conf,
        classes=classes or None,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
        save=False,
        persist=True
    )

    frame_idx = 0
    for r in stream:
        frame = getattr(r, "orig_img", None)
        if frame is None:
            frame_idx += 1
            continue

        boxes = getattr(r, "boxes", None)
        if boxes is not None:
            ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else []
            clses = boxes.cls.cpu().numpy().astype(int) if getattr(boxes, "cls", None) is not None else []
            xyxy = boxes.xyxy.cpu().numpy() if getattr(boxes, "xyxy", None) is not None else []
            for tid, cls_id, bb in zip(ids, clses, xyxy):
                if classes and (int(cls_id) not in classes):
                    continue
                x1, y1, x2, y2 = map(int, bb[:4])
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                # draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 170, 255), 2)
                cls_name = CLASS_NAMES[int(cls_id)] if 0 <= int(cls_id) < len(CLASS_NAMES) else str(int(cls_id))
                label = f"{cls_name} #{int(tid)}"  # tracker ID can be large; that's normal
                draw_label_bg(frame, label, x1, max(15, y1))

                # save record
                trajectories["records"].append({
                    "id": int(tid),
                    "frame": int(frame_idx),
                    "cx": float(cx),
                    "cy": float(cy)
                })

        writer.write(frame)
        frame_idx += 1

    writer.release()
    if not validate_nonempty_video(out_path):
        raise RuntimeError("Tracking MP4 empty/invalid.")

    json_path = out_path.with_name("trajectories.json")
    with open(json_path, "w") as jf:
        json.dump(trajectories, jf)
    return out_path, json_path

# ---------- UI ----------
st.set_page_config(page_title="Highlight & Tracking Creator", layout="centered")
st.title("Highlight & Tracking Creator")
st.markdown("""
This tool creates **short highlight videos** and can **detect & track objects** using Yolov11.
It also allows downloading the results, including a JSON file with detected object trajectories.
""")

if "video_bytes" not in st.session_state:
    st.session_state.video_bytes = None
    st.session_state.video_ext = ".mp4"

# Store processed outputs persistently across reruns
if "downloads" not in st.session_state:
    st.session_state.downloads = {}

with st.sidebar:
    sensitivity = st.slider("Change Sensitivity", 0.10, 1.00, 0.50, 0.05)
    st.markdown("""
    **Note on Change Sensitivity:**  
        - **Higher values** → more sensitive → detects smaller changes → longer highlight.  
        - **Lower values** → less sensitive → keeps only bigger changes → shorter highlight.  
        """)
    st.subheader("Object Tracking")
    run_tracking = st.checkbox("Enable YOLO11n + ByteTrack", value=True)
    st.text_input("YOLO model", value=DEFAULT_MODEL, disabled=True)
    conf = st.slider("Detection confidence", 0.05, 0.9, 0.25, 0.05)

    # Full COCO class picker; default Person
    selected = st.multiselect("Objects to track (COCO)", options=list(TRACK_OPTIONS.keys()), default=["person"])
    allowed_ids = sorted({TRACK_OPTIONS[name] for name in selected})

up = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
if up is not None:
    st.session_state.video_bytes = up.getbuffer().tobytes()
    st.session_state.video_ext = (os.path.splitext(up.name)[1] or ".mp4").lower()

if st.session_state.video_bytes:
    # Preview uploaded video 
    st.video(st.session_state.video_bytes)

process_clicked = st.button("▶️ Process", type="primary", disabled=(st.session_state.video_bytes is None))

if process_clicked:
    missing = ensure_deps()
    if missing:
        st.error("Missing packages: " + ", ".join(missing))
        st.stop()

    with st.spinner("Processing…"):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            work_in = td_path / f"input{st.session_state.video_ext}"
            work_in.write_bytes(st.session_state.video_bytes)

            # 1) Keyframes by content change (single sensitivity control)
            st.caption("Selecting keyframes…")
            kf_dir = td_path / "keyframes"
            keyframes = select_keyframes_by_change(work_in, kf_dir, sensitivity)
            if not keyframes:
                st.warning("No keyframes found. Try increasing sensitivity.")
                st.stop()

            # 2) Build short highlight
            st.caption("Building highlight (MP4)…")
            hl_path = build_highlight_from_keyframes(work_in, keyframes, td_path / "highlight")

            # 3) Optional tracking with JSON trajectories
            tracked_path = None
            traj_json_path = None
            if run_tracking:
                st.caption("Running detection & tracking…")
                try:
                    tracked_path, traj_json_path = run_tracking_to_mp4_with_json(
                        work_in, allowed_ids, conf, td_path / "annotated"
                    )
                except Exception as e:
                    st.error(f"Tracking failed: {e}")

            # ---- Persist outputs in session_state (BYTES) so reruns don't wipe them ----
            st.session_state.downloads["highlight_video"] = {
                "bytes": _read_bytes(hl_path),
                "name": hl_path.name,
                "mime": "video/mp4",
            }
            if tracked_path and Path(tracked_path).exists():
                st.session_state.downloads["tracked_video"] = {
                    "bytes": _read_bytes(tracked_path),
                    "name": Path(tracked_path).name,
                    "mime": "video/mp4",
                }
            if traj_json_path and Path(traj_json_path).exists():
                st.session_state.downloads["trajectories_json"] = {
                    "bytes": _read_bytes(Path(traj_json_path)),
                    "name": Path(traj_json_path).name,
                    "mime": "application/json",
                }

# ---- Render outputs from session_state (works across reruns & multiple downloads) ----
if "highlight_video" in st.session_state.downloads:
    st.subheader("Keyframe Highlight (MP4)")
    st.video(st.session_state.downloads["highlight_video"]["bytes"])
    st.download_button(
        "Download Highlight (MP4)",
        data=st.session_state.downloads["highlight_video"]["bytes"],
        file_name=st.session_state.downloads["highlight_video"]["name"],
        mime=st.session_state.downloads["highlight_video"]["mime"],
        key="dl_highlight",
    )

if "tracked_video" in st.session_state.downloads:
    st.subheader("Tracked Video (MP4)")
    st.video(st.session_state.downloads["tracked_video"]["bytes"])
    st.download_button(
        "Download Tracked (MP4)",
        data=st.session_state.downloads["tracked_video"]["bytes"],
        file_name=st.session_state.downloads["tracked_video"]["name"],
        mime=st.session_state.downloads["tracked_video"]["mime"],
        key="dl_tracked",
    )

if "trajectories_json" in st.session_state.downloads:
    st.download_button(
        "Download Trajectories (JSON)",
        data=st.session_state.downloads["trajectories_json"]["bytes"],
        file_name=st.session_state.downloads["trajectories_json"]["name"],
        mime=st.session_state.downloads["trajectories_json"]["mime"],
        key="dl_traj",
    )

else:
    st.info("Upload a video to get started.")

st.markdown(
"""
<hr style="margin-top: 2em; margin-bottom: 0.5em;">
<div style="text-align: center; font-size: 0.9em; color: gray;">
    Developed by <b><a href="https://www.linkedin.com/in/hananabukwaider/" target="_blank" style="text-decoration: none; color: #4da6ff;">Eng. Hanan Abu Kwaider</a></b> · 
    
    
    
</div>
""",
unsafe_allow_html=True
)
