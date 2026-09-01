import cv2
from fer.fer import FER

# Initialize emotion detector
emotion_detector = FER(mtcnn=True)
# Open webcam
camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # Detect emotions
    emotions = emotion_detector.detect_emotions(frame)

    for emotion in emotions:

        # Face bounding box
        x, y, w, h = emotion["box"]

        # Get emotion scores
        emotion_scores = emotion["emotions"]

        # Find strongest emotion
        dominant_emotion = max(
            emotion_scores,
            key=emotion_scores.get
        )

        confidence = emotion_scores[dominant_emotion]

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            2
        )

        # Display emotion
        text = f"{dominant_emotion}: {confidence:.2f}"

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # Display window
    cv2.imshow(
        "Day 8 - Emotion Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()