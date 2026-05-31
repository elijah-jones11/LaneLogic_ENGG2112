import cv2
import csv
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

# -----------------------------
# Settings
# -----------------------------

confidence_threshold = 0.2

# Choose the exact video you want to process
VIDEO_FILE = "Jim Datasets/Whole Day 3 - Tues 28th Apr.mov"

# Choose the output CSV name
OUTPUT_CSV = "Cleaned_Datasets_local/traffic_counts_Whole_Day_3_Tues_27_04_27.csv"

SAMPLE_EVERY_SECONDS = 10



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

csv_columns = [
    "video_name",
    "timestamp_seconds",
    "timestamp_hhmmss",
    "frame_number",
    "person_count",
    "bicycle_count",
    "car_count",
    "motorcycle_count",
    "bus_count",
    "truck_count",
    "traffic_light_count",
    "total_vehicle_count",
    "total_road_user_count"
]
# Load YOLO model
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")

# Check video file exists
video_path = Path(VIDEO_FILE)

if video_path.exists() == False:
    print("ERROR: Could not find video file:")
    print(video_path)
    exit()

print("Processing video:", video_path.name)

# Open video
camera = cv2.VideoCapture(str(video_path))

if camera.isOpened() == False:
    print("ERROR: Could not open video:", video_path.name)
    exit()

fps = camera.get(cv2.CAP_PROP_FPS)
total_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

if fps == 0:
    print("ERROR: FPS could not be read.")
    camera.release()
    exit()

print("FPS:", fps)
duration_seconds = total_frames / fps
print("Total frames in video:", total_frames)
print("Video duration in seconds:", round(duration_seconds, 2))
print("Estimated YOLO frames to process:", int(duration_seconds / SAMPLE_EVERY_SECONDS))

# Process one frame per second
frame_interval = int(fps)

# Create CSV
duration_seconds = total_frames / fps

with open(OUTPUT_CSV, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=csv_columns)
    writer.writeheader()

    current_second = 0

    while current_second < duration_seconds:

        frame_number = int(current_second * fps)

        # Jump directly to the frame for this second
        camera.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        success, frame = camera.read()

        if success == False:
            break

        timestamp_seconds = current_second

        hours = int(timestamp_seconds // 3600)
        minutes = int((timestamp_seconds % 3600) // 60)
        seconds = int(timestamp_seconds % 60)

        timestamp_hhmmss = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        print("Processing:", timestamp_hhmmss)

        results = model.predict(
            frame,
            conf=confidence_threshold,
            verbose=False
        )[0]

        counts = Counter()

        if results.boxes is not None:
            for box in results.boxes:

                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                if class_name in TARGET_OBJECTS:
                    counts[class_name] += 1

        total_vehicle_count = sum(counts[obj] for obj in VEHICLE_OBJECTS)
        total_road_user_count = sum(counts[obj] for obj in ROAD_USER_OBJECTS)

        row = {
            "video_name": video_path.name,
            "timestamp_seconds": timestamp_seconds,
            "timestamp_hhmmss": timestamp_hhmmss,
            "frame_number": frame_number,
            "person_count": counts["person"],
            "bicycle_count": counts["bicycle"],
            "car_count": counts["car"],
            "motorcycle_count": counts["motorcycle"],
            "bus_count": counts["bus"],
            "truck_count": counts["truck"],
            "traffic_light_count": counts["traffic light"],
            "total_vehicle_count": total_vehicle_count,
            "total_road_user_count": total_road_user_count
        }

        writer.writerow(row)

        current_second += SAMPLE_EVERY_SECONDS

camera.release()

print("\nFinished.")
print("CSV saved as:", OUTPUT_CSV)

