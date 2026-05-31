import cv2
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

# =========================================================
# SETTINGS
# =========================================================
VIDEO_FILE = "Jim Datasets/Whole Day 3 - Tues 28th Apr.mov"
OUTPUT_VIDEO = "presentation_demo/yolo_demo_40s.mp4"

# Choose the start time of the demo clip
START_TIME_SECONDS = 885        # example: 35 min into the video
CLIP_DURATION_SECONDS = 30

CONFIDENCE_THRESHOLD = 0.2

TARGET_OBJECTS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light"
]

VEHICLE_OBJECTS = [
    "car",
    "motorcycle",
    "bus",
    "truck"
]

ROAD_USER_OBJECTS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck"
]

# =========================================================
# CLASS COLOURS (OpenCV uses BGR, not RGB)
# =========================================================
CLASS_COLORS = {
    "person": (255, 255, 0),        # cyan
    "bicycle": (0, 255, 255),       # yellow
    "car": (0, 255, 0),             # green
    "motorcycle": (255, 0, 255),    # magenta / purple
    "bus": (0, 0, 255),             # red
    "truck": (255, 165, 0),         # orange
    "traffic light": (200, 200, 200) # light grey
}

# =========================================================
# LOAD MODEL
# =========================================================
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")

# =========================================================
# OPEN VIDEO
# =========================================================
video_path = Path(VIDEO_FILE)
if not video_path.exists():
    raise FileNotFoundError(f"Could not find video file: {video_path}")

camera = cv2.VideoCapture(str(video_path))
if not camera.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

fps = camera.get(cv2.CAP_PROP_FPS)
total_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps == 0:
    camera.release()
    raise RuntimeError("FPS could not be read from video.")

duration_seconds = total_frames / fps
print("Video duration:", round(duration_seconds, 2), "seconds")

# =========================================================
# PREP OUTPUT
# =========================================================
Path("presentation_demo").mkdir(parents=True, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# =========================================================
# JUMP TO START TIME
# =========================================================
start_frame = int(START_TIME_SECONDS * fps)
end_frame = int((START_TIME_SECONDS + CLIP_DURATION_SECONDS) * fps)

camera.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

print(f"Creating demo clip from {START_TIME_SECONDS}s to {START_TIME_SECONDS + CLIP_DURATION_SECONDS}s")

# =========================================================
# PROCESS FRAMES
# =========================================================
frame_idx = start_frame

while frame_idx < end_frame:
    success, frame = camera.read()

    if not success:
        break

    # Time stamp for overlay
    current_seconds = frame_idx / fps
    hours = int(current_seconds // 3600)
    minutes = int((current_seconds % 3600) // 60)
    seconds = int(current_seconds % 60)
    timestamp_hhmmss = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # YOLO prediction
    results = model.predict(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )[0]

    counts = Counter()
    annotated_frame = frame.copy()

    if results.boxes is not None:
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name in TARGET_OBJECTS:
                counts[class_name] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Pick colour for class
                color = CLASS_COLORS.get(class_name, (255, 255, 255))  # default white

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Cleaner label without confidence
                label = class_name
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

    total_vehicle_count = sum(counts[obj] for obj in VEHICLE_OBJECTS)
    total_road_user_count = sum(counts[obj] for obj in ROAD_USER_OBJECTS)

    # =====================================================
    # DRAW INFO PANEL
    # =====================================================
    panel_x1, panel_y1 = 15, 15
    panel_x2, panel_y2 = 320, 235

    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
    alpha = 0.45
    annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)

    lines = [
        f"Time: {timestamp_hhmmss}",
        f"Person: {counts['person']}",
        f"Bicycle: {counts['bicycle']}",
        f"Car: {counts['car']}",
        f"Motorcycle: {counts['motorcycle']}",
        f"Bus: {counts['bus']}",
        f"Truck: {counts['truck']}",
        f"Traffic light: {counts['traffic light']}",
        f"Total vehicles: {total_vehicle_count}",
        f"Total road users: {total_road_user_count}",
    ]

    y_text = 40
    for line in lines:
        cv2.putText(
            annotated_frame,
            line,
            (30, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )
        y_text += 19

    writer.write(annotated_frame)

    # Optional preview while generating
    cv2.imshow("YOLO Demo Preview", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_idx += 1

# =========================================================
# CLEAN UP
# =========================================================
camera.release()
writer.release()
cv2.destroyAllWindows()

print("\nFinished.")
print("Saved demo video to:", OUTPUT_VIDEO)

