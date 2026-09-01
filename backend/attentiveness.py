import cv2
import mediapipe as mp
import math

# -----------------------------
# MediaPipe Face Mesh
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

# -----------------------------
# Landmark indexes
# -----------------------------

# Eyes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

# -----------------------------
# Thresholds
# -----------------------------

EAR_THRESHOLD = 0.22
MOUTH_THRESHOLD = 0.30

# -----------------------------
# Functions
# -----------------------------


def distance(point1, point2):
    return math.dist(point1, point2)


def calculate_ear(eye_points):

    vertical_1 = distance(
        eye_points[1],
        eye_points[5]
    )

    vertical_2 = distance(
        eye_points[2],
        eye_points[4]
    )

    horizontal = distance(
        eye_points[0],
        eye_points[3]
    )

    return (vertical_1 + vertical_2) / (
        2 * horizontal
    )


def calculate_mouth_ratio(
    landmarks,
    width,
    height
):

    upper = landmarks[UPPER_LIP]
    lower = landmarks[LOWER_LIP]

    left = landmarks[LEFT_MOUTH]
    right = landmarks[RIGHT_MOUTH]

    upper_point = (
        int(upper.x * width),
        int(upper.y * height)
    )

    lower_point = (
        int(lower.x * width),
        int(lower.y * height)
    )

    left_point = (
        int(left.x * width),
        int(left.y * height)
    )

    right_point = (
        int(right.x * width),
        int(right.y * height)
    )

    vertical = distance(
        upper_point,
        lower_point
    )

    horizontal = distance(
        left_point,
        right_point
    )

    return vertical / horizontal


# -----------------------------
# Open webcam
# -----------------------------

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

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Detect landmarks
        results = face_mesh.process(rgb)

        # Default score
        attention_score = 100

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                h, w, _ = frame.shape

                landmarks = face_landmarks.landmark

                # -----------------------------
                # Eye points
                # -----------------------------

                left_eye = []

                for index in LEFT_EYE:

                    point = landmarks[index]

                    left_eye.append(
                        (
                            int(point.x * w),
                            int(point.y * h)
                        )
                    )

                right_eye = []

                for index in RIGHT_EYE:

                    point = landmarks[index]

                    right_eye.append(
                        (
                            int(point.x * w),
                            int(point.y * h)
                        )
                    )

                # -----------------------------
                # Calculate EAR
                # -----------------------------

                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)

                ear = (
                    left_ear + right_ear
                ) / 2

                # -----------------------------
                # Eye status
                # -----------------------------

                if ear < EAR_THRESHOLD:

                    eye_status = "Eyes Closed"

                    attention_score -= 20

                else:

                    eye_status = "Eyes Open"

                # -----------------------------
                # Mouth status
                # -----------------------------

                mouth_ratio = calculate_mouth_ratio(
                    landmarks,
                    w,
                    h
                )

                if mouth_ratio > MOUTH_THRESHOLD:

                    mouth_status = "Mouth Open"

                    attention_score -= 10

                else:

                    mouth_status = "Mouth Closed"

                # -----------------------------
                # Head direction
                # -----------------------------

                nose = landmarks[1]
                left_face = landmarks[234]
                right_face = landmarks[454]

                nose_x = nose.x
                left_x = left_face.x
                right_x = right_face.x

                face_center = (
                    left_x + right_x
                ) / 2

                difference = nose_x - face_center

                if difference < -0.08:

                    head_status = "Looking Left"

                    attention_score -= 20

                elif difference > 0.08:

                    head_status = "Looking Right"

                    attention_score -= 20

                else:

                    head_status = "Looking Forward"

                # -----------------------------
                # Keep score between 0 and 100
                # -----------------------------

                attention_score = max(
                    0,
                    min(100, attention_score)
                )

                # -----------------------------
                # Display information
                # -----------------------------

                cv2.putText(
                    frame,
                    f"Attention: {attention_score}%",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    eye_status,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    mouth_status,
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    head_status,
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

        else:

            cv2.putText(
                frame,
                "No Face Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        # Show webcam
        cv2.imshow(
            "Day 9 - Attentiveness Detection",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


camera.release()
cv2.destroyAllWindows()