import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Mouth landmark points
UPPER_LIP = 13
LOWER_LIP = 14

LEFT_MOUTH = 61
RIGHT_MOUTH = 291

# Thresholds
MOUTH_OPEN_THRESHOLD = 0.30
YAWN_FRAMES = 10

mouth_open_frames = 0
yawn_count = 0
yawn_detected = False


def calculate_mouth_ratio(landmarks, width, height):
    # Upper lip
    upper = landmarks[UPPER_LIP]
    upper_point = (
        int(upper.x * width),
        int(upper.y * height)
    )

    # Lower lip
    lower = landmarks[LOWER_LIP]
    lower_point = (
        int(lower.x * width),
        int(lower.y * height)
    )

    # Left corner
    left = landmarks[LEFT_MOUTH]
    left_point = (
        int(left.x * width),
        int(left.y * height)
    )

    # Right corner
    right = landmarks[RIGHT_MOUTH]
    right_point = (
        int(right.x * width),
        int(right.y * height)
    )

    # Vertical mouth distance
    vertical_distance = math.dist(
        upper_point,
        lower_point
    )

    # Horizontal mouth distance
    horizontal_distance = math.dist(
        left_point,
        right_point
    )

    # Mouth Opening Ratio
    ratio = vertical_distance / horizontal_distance

    return ratio, upper_point, lower_point, left_point, right_point


# Open webcam
camera = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:

        success, frame = camera.read()

        if not success:
            break

        # Mirror effect
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Detect face landmarks
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                height, width, _ = frame.shape

                landmarks = face_landmarks.landmark

                # Calculate mouth ratio
                ratio, upper, lower, left, right = calculate_mouth_ratio(
                    landmarks,
                    width,
                    height
                )

                # Draw mouth points
                cv2.circle(frame, upper, 4, (255, 255, 255), -1)
                cv2.circle(frame, lower, 4, (255, 255, 255), -1)
                cv2.circle(frame, left, 4, (255, 255, 255), -1)
                cv2.circle(frame, right, 4, (255, 255, 255), -1)

                # Check mouth status
                if ratio > MOUTH_OPEN_THRESHOLD:

                    mouth_open_frames += 1

                    mouth_status = "Mouth Open"

                else:

                    mouth_status = "Mouth Closed"

                    # Detect yawn after mouth remains open
                    if mouth_open_frames >= YAWN_FRAMES:

                        yawn_count += 1
                        yawn_detected = True

                    mouth_open_frames = 0

                # Display yawn message
                if yawn_detected:

                    cv2.putText(
                        frame,
                        "YAWN DETECTED!",
                        (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2
                    )

                    # Reset after displaying
                    yawn_detected = False

                # Display mouth ratio
                cv2.putText(
                    frame,
                    f"Mouth Ratio: {ratio:.2f}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                # Display mouth status
                cv2.putText(
                    frame,
                    mouth_status,
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                # Display yawn count
                cv2.putText(
                    frame,
                    f"Yawns: {yawn_count}",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

        # Show webcam
        cv2.imshow(
            "Day 6 - Mouth & Yawn Detection",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


camera.release()
cv2.destroyAllWindows()