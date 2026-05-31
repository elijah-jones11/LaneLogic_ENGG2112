import cv2
from ultralytics import YOLO

# Variables:
confidence_threshold = 0.2  # how confident the model needs to be to show the box
TARGET_OBJECTS = ["person", "bicycle", "motorcycle", "car", "bus", "truck", "traffic light", "handbag", "bottle", "chair", "couch", "potted plant", "dining table", "laptop", "tv", "mouse", "remote", "keyboard", "cell phone", "book", "scissors"]  
# ^^^ can change the list to any variables; truck, car, bus, etc... ^^^

BOX_THICKNESS = 2
BOX_COLOURS = {
    "person": (0, 0, 255),            # Red
    "bicycle": (0, 165, 255),         # Orange
    "car": (235, 206, 135),           # Sky Blue
    "bus": (0, 255, 255),             # Yellow
    "truck": (128, 0, 128),           # Purple
    "traffic light": (200, 200, 200)  # Light Grey
}
DEFAULT_BOX_COLOUR = (0, 255, 0)      # Green

BOX_TEXT_SIZE = (0.8)
BOX_TEXT_THICKNESS = (2)

COUNT_TEXT_COLOUR = (0, 0, 255)       # Red
COUNT_TEXT_SIZE = (1.2)
COUNT_TEXT_THICKNESS = (2)

# Loading the YOLO Model (all data from the ultralytics module):
print("LOADING YOLO MODEL...")
model = YOLO("yolov8n.pt")   # freesourced???


# For fun --> show what the YOLO Model can detect (what it has data on):
print("\nYOLO can detect the following objects:\n")

for class_id in model.names:
    class_name = model.names[class_id]
    print(class_id, "->", class_name)

print("\nSTARTING WEBCAM...\n")

# Opening the webcam (cv2 module)
#camera = cv2.VideoCapture("Datasets/traffic_1.mp4")
camera = cv2.VideoCapture(0)

if camera.isOpened() == False:
    print("ERROR: Could not open webcam.")
    print("Try changing VideoCapture(0) to VideoCapture(1).")
    exit()


# Being reading information from the webcam:
while True:

    # Read one frame from the webcam
    success, frame = camera.read()

    if success == False:
        print("Failed to read frame from webcam.")
        break

    # Run the YOLO detection on the frame:
    # If you want to output constant signals in terminal, replace the line below with this:
        # results_list = model(frame)
        # results = results_list[0]
    results = model.predict(frame, conf=confidence_threshold, verbose=False)[0]



    counts = {}

    # Create boxes and loop them:
    if results.boxes is not None:

        for box in results.boxes:

            # Get class ID (number)
            class_id = int(box.cls[0])

            # Convert class ID to readable name
            class_name = model.names[class_id]

            # Only continue if it's our target object
            if class_name in TARGET_OBJECTS:

                if class_name not in counts:
                    counts[class_name] = 0

                counts[class_name] += 1

                # Box coordinates
                x1 = int(box.xyxy[0][0])
                y1 = int(box.xyxy[0][1])
                x2 = int(box.xyxy[0][2])
                y2 = int(box.xyxy[0][3])

                confidence = float(box.conf[0])

                # Colour rectangles:
                if class_name in BOX_COLOURS:
                    box_colour = BOX_COLOURS[class_name]
                else:
                    box_colour = DEFAULT_BOX_COLOUR

                # Draw rectangle:
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, BOX_THICKNESS)
                
                # Write label text
                text = class_name + " " + str(round(confidence, 2))

                cv2.putText(frame,
                            text,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            BOX_TEXT_SIZE,
                            box_colour,
                            BOX_TEXT_THICKNESS)

    # Display how many total boxes are detected:

   
   
    display_text = ""

    y_position = 30
    for obj in counts:
        count_text = obj + ": " + str(counts[obj])

        # Get the same colour as the box
        if obj in BOX_COLOURS:
            text_colour = BOX_COLOURS[obj]
        else:
            text_colour = DEFAULT_BOX_COLOUR

        cv2.putText(frame,
                    count_text,
                    (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    COUNT_TEXT_SIZE,
                    text_colour,
                    COUNT_TEXT_THICKNESS)
        y_position += 40
    


    # Show the frame on the computer screen:
    cv2.imshow("YOLO Object Detector", frame)

    # QUIT PROCESS --> user can press 'Q':
    key = cv2.waitKey(1)

    if key == ord("q") or key == ord("Q"):
        break

# Cleaning:
camera.release()
cv2.destroyAllWindows()




