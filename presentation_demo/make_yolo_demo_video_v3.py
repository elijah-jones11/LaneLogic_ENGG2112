import cv2
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

# =========================================================
# SETTINGS
# =========================================================
VIDEO_FILE = "Jim Datasets/Whole Day 2 - Mon 27th Apr.mov"
OUTPUT_VIDEO = "presentation_demo/yolo_demo_20s_7.mp4"

# Choose the start time of the demo clip
START_TIME_SECONDS = 9625
CLIP_DURATION_SECONDS = 20

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
# BOX / TEXT STYLING
# OpenCV uses BGR
# =========================================================
BOX_THICKNESS = 2

BOX_COLOURS = {
    "person": (255, 255, 0),         # Cyan
    "bicycle": (0, 165, 255),        # Orange
    "car": (0, 255, 0),              # Green
    "motorcycle": (255, 0, 255),     # Magenta
    "bus": (0, 0, 255),              # Red
    "truck": (128, 0, 128),          # Purple
    "traffic light": (180, 180, 180) # Light Grey
}
DEFAULT_BOX_COLOUR = (0, 255, 255)   # Yellow-ish fallback

BOX_TEXT_SIZE = 0.8
BOX_TEXT_THICKNESS = 2

COUNT_TEXT_SIZE = 0.75
COUNT_TEXT_THICKNESS = 2

# =========================================================
# LOAD MODEL
# =========================================================
print("LOADING YOLO MODEL...")
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

    # Run YOLO
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    counts = Counter()

    # Draw boxes directly on frame
    if results.boxes is not None:
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name in TARGET_OBJECTS:
                counts[class_name] += 1

                x1 = int(box.xyxy[0][0])
                y1 = int(box.xyxy[0][1])
                x2 = int(box.xyxy[0][2])
                y2 = int(box.xyxy[0][3])

                confidence = float(box.conf[0])

                # Pick box colour
                box_colour = BOX_COLOURS.get(class_name, DEFAULT_BOX_COLOUR)

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, BOX_THICKNESS)

                # Label text
                text = class_name + " " + str(round(confidence, 2))

                cv2.putText(
                    frame,
                    text,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    BOX_TEXT_SIZE,
                    box_colour,
                    BOX_TEXT_THICKNESS
                )

    total_vehicle_count = sum(counts[obj] for obj in VEHICLE_OBJECTS)
    total_road_user_count = sum(counts[obj] for obj in ROAD_USER_OBJECTS)

    # =====================================================
    # DRAW INFO PANEL BACKGROUND
    # =====================================================
    panel_x1, panel_y1 = 10, 10
    panel_x2, panel_y2 = 460, 380

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
    alpha = 0.40
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Time at top in white
    cv2.putText(
        frame,
        f"Time: {timestamp_hhmmss}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    # Display counts using same colour as boxes
    display_items = [
        ("person", counts["person"]),
        ("bicycle", counts["bicycle"]),
        ("car", counts["car"]),
        ("motorcycle", counts["motorcycle"]),
        ("bus", counts["bus"]),
        ("truck", counts["truck"]),
        ("traffic light", counts["traffic light"]),
    ]

    y_position = 85
    line_spacing = 33

    for obj, value in display_items:
        count_text = obj + ": " + str(value)
        text_colour = BOX_COLOURS.get(obj, DEFAULT_BOX_COLOUR)

        cv2.putText(
            frame,
            count_text,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            COUNT_TEXT_SIZE,
            text_colour,
            COUNT_TEXT_THICKNESS
        )
        y_position += line_spacing

    # Put totals BELOW the last class row with a proper gap
    totals_y1 = y_position + 12
    totals_y2 = totals_y1 + 30

    cv2.putText(
        frame,
        f"Total vehicles: {total_vehicle_count}",
        (20, totals_y1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Total road users: {total_road_user_count}",
        (20, totals_y2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2
    )

    writer.write(frame)

    # Optional preview
    cv2.imshow("YOLO Demo Preview", frame)
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