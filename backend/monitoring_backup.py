import cv2
import mediapipe as mp
import math
import csv
import os
import time
from collections import deque

# IMPORTANT:
# Your installed FER version uses this import
from fer.fer import FER


# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = "attention_data.csv"

# Save data every 5 seconds
LOG_INTERVAL = 5

# Attention smoothing
SMOOTHING_FRAMES = 10

# Eye detection
EAR_THRESHOLD = 0.22

# Mouth detection
MOUTH_THRESHOLD = 0.30

# Blink detection
BLINK_FRAMES = 2

# Yawn detection
YAWN_FRAMES = 10

# FER is heavy, so don't run it every frame
EMOTION_INTERVAL = 15


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh


# ============================================================
# EMOTION DETECTOR
# ============================================================

print("Loading emotion detector...")

emotion_detector = FER(mtcnn=False)

print("Emotion detector loaded.")


# ============================================================
# LANDMARKS
# ============================================================

# Left eye
LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

# Right eye
RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]

# Mouth
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291


# ============================================================
# COUNTERS
# ============================================================

blink_count = 0
yawn_count = 0

closed_frames = 0
mouth_open_frames = 0

score_history = deque(
    maxlen=SMOOTHING_FRAMES
)

last_log_time = time.time()

frame_count = 0


# ============================================================
# PREVIOUS EMOTION
# ============================================================

last_emotion = "Unknown"

last_emotion_confidence = 0.0


# ============================================================
# FUNCTIONS
# ============================================================

def distance(point1, point2):
    """
    Calculate distance between two points.
    """
    return math.dist(point1, point2)


# ------------------------------------------------------------
# Eye Aspect Ratio
# ------------------------------------------------------------

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

    ear = (
        vertical_1 + vertical_2
    ) / (2 * horizontal)

    return ear


# ------------------------------------------------------------
# Mouth Ratio
# ------------------------------------------------------------

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
# CREATE / FIX CSV
# ============================================================

CSV_HEADER = [
    "Time",
    "Attention",
    "Emotion",
    "Confidence",
    "Eyes",
    "Mouth",
    "Blinks",
    "Yawns",
    "Head"
]


# If CSV doesn't exist, create it
if not os.path.exists(CSV_FILE):

    with open(
        CSV_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            CSV_HEADER
        )


else:

    # Check existing CSV header
    try:

        with open(
            CSV_FILE,
            "r",
            newline=""
        ) as file:

            reader = csv.reader(file)

            old_header = next(
                reader,
                []
            )


        # If old CSV has wrong columns,
        # create a new file with correct header
        if old_header != CSV_HEADER:

            print(
                "Old CSV format detected."
            )

            print(
                "Creating new CSV format..."
            )

            # Backup old CSV
            backup_file = (
                "attention_data_old.csv"
            )

            try:

                os.rename(
                    CSV_FILE,
                    backup_file
                )

                print(
                    "Old CSV backed up as:",
                    backup_file
                )

            except Exception:

                pass


            with open(
                CSV_FILE,
                "w",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow(
                    CSV_HEADER
                )


    except Exception:

        with open(
            CSV_FILE,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                CSV_HEADER
            )


# ============================================================
# WEBCAM
# ============================================================

print("Opening camera...")

camera = cv2.VideoCapture(0)


# Camera resolution
camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


# Camera FPS
camera.set(
    cv2.CAP_PROP_FPS,
    20
)


# Reduce camera buffering
camera.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not camera.isOpened():

    print(
        "ERROR: Could not open camera."
    )

    exit()


print(
    "Camera started."
)

print(
    "Press Q to stop monitoring."
)


# ============================================================
# FACE MESH
# ============================================================

with mp_face_mesh.FaceMesh(

    max_num_faces=1,

    # Do NOT draw landmarks
    refine_landmarks=False,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5

) as face_mesh:


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:


        # ----------------------------------------------------
        # READ CAMERA
        # ----------------------------------------------------

        success, frame = camera.read()


        if not success:

            print(
                "Could not read camera frame."
            )

            break


        frame_count += 1


        # ----------------------------------------------------
        # MIRROR CAMERA
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        frame = cv2.resize(
            frame,
            (640, 480)
        )


        height, width, _ = frame.shape


        # ====================================================
        # DEFAULT VALUES
        # ====================================================

        attention_score = 0

        dominant_emotion = last_emotion

        emotion_confidence = (
            last_emotion_confidence
        )

        eye_status = "Unknown"

        mouth_status = "Unknown"

        head_status = "Unknown"


        # ====================================================
        # EMOTION DETECTION
        # ====================================================

        if (
            frame_count %
            EMOTION_INTERVAL
            == 0
        ):

            try:

                emotions = (
                    emotion_detector.detect_emotions(
                        frame
                    )
                )


                if emotions:

                    emotion_scores = (
                        emotions[0]["emotions"]
                    )


                    dominant_emotion = max(
                        emotion_scores,
                        key=emotion_scores.get
                    )


                    emotion_confidence = (
                        emotion_scores[
                            dominant_emotion
                        ]
                    )


                    # Save for next frames
                    last_emotion = (
                        dominant_emotion
                    )

                    last_emotion_confidence = (
                        emotion_confidence
                    )


            except Exception as e:

                print(
                    "Emotion detection error:",
                    e
                )


        # ====================================================
        # MEDIAPIPE FACE LANDMARKS
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        results = face_mesh.process(
            rgb
        )


        # ====================================================
        # FACE FOUND
        # ====================================================

        if results.multi_face_landmarks:

            face = (
                results.multi_face_landmarks[0]
            )

            landmarks = face.landmark


            # =================================================
            # START ATTENTION SCORE
            # =================================================

            raw_score = 100


            # =================================================
            # EYES
            # =================================================

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
            left_ear = calculate_ear(
                left_eye
            )

            right_ear = calculate_ear(
                right_eye
            )


            ear = (
                left_ear +
                right_ear
            ) / 2


            # -------------------------------------------------
            # EYE STATUS
            # -------------------------------------------------

            if ear < EAR_THRESHOLD:

                closed_frames += 1

                eye_status = (
                    "Eyes Closed"
                )

                raw_score -= 20


            else:

                # Count blink when eyes reopen
                if (
                    closed_frames >=
                    BLINK_FRAMES
                ):

                    blink_count += 1


                closed_frames = 0

                eye_status = (
                    "Eyes Open"
                )


            # =================================================
            # MOUTH
            # =================================================

            mouth_ratio = (
                calculate_mouth_ratio(
                    landmarks,
                    width,
                    height
                )
            )


            if (
                mouth_ratio >
                MOUTH_THRESHOLD
            ):

                mouth_open_frames += 1

                mouth_status = (
                    "Mouth Open"
                )

                raw_score -= 10


            else:

                # Count yawn after mouth
                # has been open long enough
                if (
                    mouth_open_frames >=
                    YAWN_FRAMES
                ):

                    yawn_count += 1


                mouth_open_frames = 0

                mouth_status = (
                    "Mouth Closed"
                )


            # =================================================
            # HEAD DIRECTION
            # =================================================

            nose = landmarks[1]

            left_face = landmarks[234]

            right_face = landmarks[454]


            face_center = (
                left_face.x +
                right_face.x
            ) / 2


            difference = (
                nose.x -
                face_center
            )


            if difference < -0.08:

                head_status = "Left"

                raw_score -= 20


            elif difference > 0.08:

                head_status = "Right"

                raw_score -= 20


            else:

                head_status = "Forward"


            # =================================================
            # LIMIT ATTENTION SCORE
            # =================================================

            raw_score = max(
                0,
                min(
                    100,
                    raw_score
                )
            )


            # =================================================
            # SMOOTH ATTENTION
            # =================================================

            score_history.append(
                raw_score
            )


            attention_score = (
                sum(score_history)
                /
                len(score_history)
            )


        # ====================================================
        # NO FACE
        # ====================================================

        else:

            attention_score = 0

            eye_status = "No Face"

            mouth_status = "No Face"

            head_status = "No Face"


        # ====================================================
        # ATTENTION STATUS
        # ====================================================

        if attention_score >= 80:

            attention_status = (
                "Attentive"
            )

        elif attention_score >= 50:

            attention_status = (
                "Moderate"
            )

        else:

            attention_status = (
                "Distracted"
            )


        # ====================================================
        # CONFIDENCE AS PERCENTAGE
        # ====================================================

        confidence_percentage = (
            emotion_confidence * 100
        )


        # ====================================================
        # SAVE DATA EVERY 5 SECONDS
        # ====================================================

        current_time = time.time()


        if (
            current_time -
            last_log_time
            >= LOG_INTERVAL
        ):


            with open(
                CSV_FILE,
                "a",
                newline=""
            ) as file:


                writer = csv.writer(
                    file
                )


                writer.writerow([

                    time.strftime(
                        "%H:%M:%S"
                    ),

                    round(
                        attention_score
                    ),

                    dominant_emotion,

                    round(
                        confidence_percentage,
                        1
                    ),

                    eye_status,

                    mouth_status,

                    blink_count,

                    yawn_count,

                    head_status

                ])


            # ------------------------------------------------
            # TERMINAL OUTPUT
            # ------------------------------------------------

            print(
                "Data saved:",
                round(
                    attention_score
                ),
                dominant_emotion,
                f"{confidence_percentage:.1f}%",
                eye_status,
                mouth_status,
                blink_count,
                yawn_count,
                head_status
            )


            last_log_time = (
                current_time
            )


        # ====================================================
        # DISPLAY ON CAMERA
        # ====================================================

        cv2.putText(

            frame,

            f"Attention: "
            f"{round(attention_score)}%",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Status: "
            f"{attention_status}",

            (20, 75),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Emotion: "
            f"{dominant_emotion}",

            (20, 110),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Confidence: "
            f"{confidence_percentage:.1f}%",

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

            f"Head: "
            f"{head_status}",

            (20, 250),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Blinks: "
            f"{blink_count}",

            (20, 285),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Yawns: "
            f"{yawn_count}",

            (20, 320),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2

        )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            "Live Facial Emotion & Attention Monitoring",
            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        if (
            cv2.waitKey(1) &
            0xFF
        ) == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()


print(
    "Monitoring stopped."
)

print(
    "Data saved in:",
    CSV_FILE
)