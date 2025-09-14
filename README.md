# Highlight & Tracking Creator
A **Streamlit** app for automatically generating **video highlights** and **tracking objects** using **YOLOv11 + ByteTrack**.

## ✨ Features
- **Automatic Highlights**  
  Extracts keyframes by detecting major scene changes, then stitches them into a short fast-forward style video.
- **Object Detection & Tracking**  
  Track selected objects (people, vehicles, etc.) with YOLOv11 model and ByteTrack.
- **Trajectory Export**  
  Saves tracked object trajectories as JSON (`id`, `frame`, `centroid_x`, `centroid_y`).
- **User-Friendly UI**  
  Adjustable change sensitivity for highlight generation, object selection via dropdown, and one-click downloads.

## 🎬 Demo
**Downloads:**
- [Highlight video](assets/demo-highlight.mp4)
- [Tracked video](assets/demo-annotated.mp4)
- [Trajectories JSON](assets/demo-trajectories.json)
