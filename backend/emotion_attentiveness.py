import cv2
import mediapipe as mp
import numpy as np
import math

# FER
from fer.fer import FER


# ============================================================
# INITIALIZATION
# ============================================================

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Emotion detector
emotion_detector = FER(mtcnn=True)

# Webcam
camera = cv2.VideoCapture(0)


# ============================================================
# LANDMARKS
# ============================================================

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmarks
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291


# ============================================================
# THRESHOLDS
# ============================================================

EAR_THRESHOLD = 0.22
MOUTH_THRESHOLD = 0.30

# Number of frames eyes must remain closed
BLINK_FRAMES = 2

# Number of frames mouth must remain open
YAWN_FRAMES = 10


# ============================================================
# COUNTERS
# ============================================================

blink_count = 0
yawn_count = 0

closed_frames = 0
mouth_open_frames = 0


# ============================================================
# FUNCTIONS
# ============================================================

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

    if horizontal == 0:
        return 0

    return (
        vertical_1 + vertical_2
    ) / (2 * horizontal)


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

    if horizontal == 0:
        return 0

    return vertical / horizontal


# ============================================================
# FACE MESH
# ============================================================

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

        height, width, _ = frame.shape

        # ----------------------------------------------------
        # EMOTION DETECTION
        # ----------------------------------------------------

        emotions = emotion_detector.detect_emotions(frame)

        dominant_emotion = "Unknown"
        emotion_confidence = 0

        if emotions:

            emotion_scores = emotions[0]["emotions"]

            dominant_emotion = max(
                emotion_scores,
                key=emotion_scores.get
            )

            emotion_confidence = emotion_scores[
                dominant_emotion
            ]

        # ----------------------------------------------------
        # FACE LANDMARK DETECTION
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb)

        # Start with 100
        attention_score = 100

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                landmarks = face_landmarks.landmark

                # ==================================================
                # EYE DETECTION
                # ==================================================

                left_eye = []

                for index in LEFT_EYE:

                    point = landmarks[index]

                    left_eye.append(
                        (
                            int(point.x * width),
                            int(point.y * height)
                        )
                    )

                right_eye = []

                for index in RIGHT_EYE:

                    point = landmarks[index]

                    right_eye.append(
                        (
                            int(point.x * width),
                            int(point.y * height)
                        )
                    )

                # Calculate EAR
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)

                ear = (
                    left_ear + right_ear
                ) / 2

                # ------------------------------------------------
                # Eye status
                # ------------------------------------------------

                if ear < EAR_THRESHOLD:

                    closed_frames += 1

                    eye_status = "Eyes Closed"

                    attention_score -= 20

                else:

                    if closed_frames >= BLINK_FRAMES:
                        blink_count += 1

                    closed_frames = 0

                    eye_status = "Eyes Open"

                # ==================================================
                # MOUTH DETECTION
                # ==================================================

                mouth_ratio = calculate_mouth_ratio(
                    landmarks,
                    width,
                    height
                )

                if mouth_ratio > MOUTH_THRESHOLD:

                    mouth_open_frames += 1

                    mouth_status = "Mouth Open"

                else:

                    if mouth_open_frames >= YAWN_FRAMES:

                        yawn_count += 1

                    mouth_open_frames = 0

                    mouth_status = "Mouth Closed"

                # Reduce attention for open mouth
                if mouth_ratio > MOUTH_THRESHOLD:
                    attention_score -= 10

                # ==================================================
                # HEAD DIRECTION
                # ==================================================

                nose = landmarks[1]
                left_face = landmarks[234]
                right_face = landmarks[454]

                nose_x = nose.x
                left_x = left_face.x
                right_x = right_face.x

                face_center = (
                    left_x + right_x
                ) / 2

                head_difference = (
                    nose_x - face_center
                )

                if head_difference < -0.08:

                    head_status = "Looking Left"

                    attention_score -= 20

                elif head_difference > 0.08:

                    head_status = "Looking Right"

                    attention_score -= 20

                else:

                    head_status = "Looking Forward"

                # ==================================================
                # LIMIT SCORE
                # ==================================================

                attention_score = max(
                    0,
                    min(100, attention_score)
                )

                # ==================================================
                # ATTENTION STATUS
                # ==================================================

                if attention_score >= 80:

                    attention_status = "Attentive"

                elif attention_score >= 50:

                    attention_status = "Moderately Attentive"

                else:

                    attention_status = "Distracted"

                # ==================================================
                # DRAW FACE BOX
                # ==================================================

                x_coordinates = [
                    int(point.x * width)
                    for point in landmarks
                ]

                y_coordinates = [
                    int(point.y * height)
                    for point in landmarks
                ]

                x_min = max(0, min(x_coordinates))
                y_min = max(0, min(y_coordinates))
                x_max = min(width, max(x_coordinates))
                y_max = min(height, max(y_coordinates))

                cv2.rectangle(
                    frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    (255, 255, 255),
                    2
                )

        else:

            attention_score = 0
            attention_status = "No Face"

            eye_status = "Unknown"
            mouth_status = "Unknown"
            head_status = "Unknown"

        # ========================================================
        # DISPLAY INFORMATION
        # ========================================================

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
            f"Status: {attention_status}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Emotion: {dominant_emotion}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Emotion Confidence: {emotion_confidence:.2f}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            eye_status,
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            mouth_status,
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            head_status,
            (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Blinks: {blink_count}",
            (20, 285),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Yawns: {yawn_count}",
            (20, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ========================================================
        # SHOW WINDOW
        # ========================================================

        cv2.imshow(
            "Live Facial Emotion & Attentiveness",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ============================================================
# CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()