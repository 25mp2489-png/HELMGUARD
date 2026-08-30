from ultralytics import YOLO
import cv2
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(BASE_DIR, "model", "plate_best.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "uploads", "no_helmet.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "plate_best_test.jpg")

print("Loading plate model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")
print("Classes:", model.names)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

# Take frame at 2 seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 2))

success, frame = cap.read()
cap.release()

if not success:
    print("Could not read video frame.")
    exit()

print("Frame extracted successfully!")

# Detect plate
results = model.predict(
    source=frame,
    conf=0.10,
    save=False
)

result = results[0]

print("\nNumber plate detections:")

if result.boxes is None or len(result.boxes) == 0:

    print("❌ NO NUMBER PLATE DETECTED")

else:

    print("✅ Number plates detected:", len(result.boxes))

    for box in result.boxes:

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        print(
            "Class:",
            class_name,
            "| Confidence:",
            round(confidence * 100, 2),
            "%"
        )

    # Draw bounding box
    annotated = result.plot()

    cv2.imwrite(
        OUTPUT_PATH,
        annotated
    )

    print("\nAnnotated image saved at:")
    print(OUTPUT_PATH)