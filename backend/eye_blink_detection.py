import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Eye landmark indexes
# Left eye
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# Right eye
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Blink settings
EAR_THRESHOLD = 0.22
CONSECUTIVE_FRAMES = 2

blink_count = 0
closed_frames = 0


def calculate_ear(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR)
    """

    # Vertical distances
    vertical_1 = math.dist(eye_points[1], eye_points[5])
    vertical_2 = math.dist(eye_points[2], eye_points[4])

    # Horizontal distance
    horizontal = math.dist(eye_points[0], eye_points[3])

    # EAR formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


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

        # Flip camera for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process face
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                h, w, _ = frame.shape

                # Get left eye points
                left_eye_points = []

                for index in LEFT_EYE:
                    landmark = face_landmarks.landmark[index]

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    left_eye_points.append((x, y))

                # Get right eye points
                right_eye_points = []

                for index in RIGHT_EYE:
                    landmark = face_landmarks.landmark[index]

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    right_eye_points.append((x, y))

                # Calculate EAR
                left_ear = calculate_ear(left_eye_points)
                right_ear = calculate_ear(right_eye_points)

                # Average EAR
                ear = (left_ear + right_ear) / 2.0

                # Draw eye points
                for point in left_eye_points:
                    cv2.circle(frame, point, 2, (255, 255, 255), -1)

                for point in right_eye_points:
                    cv2.circle(frame, point, 2, (255, 255, 255), -1)

                # Check eyes
                if ear < EAR_THRESHOLD:

                    closed_frames += 1

                    eye_status = "Eyes Closed"

                else:

                    if closed_frames >= CONSECUTIVE_FRAMES:
                        blink_count += 1

                    closed_frames = 0

                    eye_status = "Eyes Open"

                # Display EAR
                cv2.putText(
                    frame,
                    f"EAR: {ear:.2f}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                # Display eye status
                cv2.putText(
                    frame,
                    eye_status,
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                # Display blink count
                cv2.putText(
                    frame,
                    f"Blinks: {blink_count}",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

        # Show webcam
        cv2.imshow("Day 5 - Eye & Blink Detection", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()