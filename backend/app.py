# ============================================================
# LIVE FACIAL EMOTION & ATTENTION MONITOR
# STABLE CAMERA + LOW LAG VERSION
# ============================================================

import streamlit as st
import cv2
import mediapipe as mp
import math
import csv
import os
import time
import threading
import pandas as pd

from collections import deque
from fer.fer import FER
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from streamlit_autorefresh import st_autorefresh


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Facial Emotion & Attentiveness Monitor",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = "attention_data.csv"

CAMERA_WIDTH = 480
CAMERA_HEIGHT = 360

CAMERA_FPS = 15

LOG_INTERVAL = 5

SMOOTHING_FRAMES = 8

EAR_THRESHOLD = 0.21
BLINK_FRAMES = 2

MOUTH_THRESHOLD = 0.30
YAWN_FRAMES = 10

EMOTION_INTERVAL_SECONDS = 2.0


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh


# ============================================================
# FACIAL LANDMARKS
# ============================================================

LEFT_EYE = [
    33, 160, 158,
    133, 153, 144
]

RIGHT_EYE = [
    362, 385, 387,
    263, 373, 380
]

UPPER_LIP = 13
LOWER_LIP = 14

LEFT_MOUTH = 61
RIGHT_MOUTH = 291


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def distance(p1, p2):

    return math.dist(p1, p2)


def calculate_ear(points):

    vertical_1 = distance(
        points[1],
        points[5]
    )

    vertical_2 = distance(
        points[2],
        points[4]
    )

    horizontal = distance(
        points[0],
        points[3]
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
# VIDEO PROCESSOR
# ============================================================

class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        # ====================================================
        # MEDIAPIPE
        # ====================================================

        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ====================================================
        # FER
        # ====================================================

        self.emotion_detector = FER(
            mtcnn=False
        )

        # ====================================================
        # COUNTERS
        # ====================================================

        self.blink_count = 0
        self.yawn_count = 0

        self.closed_frames = 0
        self.mouth_open_frames = 0

        # ====================================================
        # ATTENTION
        # ====================================================

        self.score_history = deque(
            maxlen=SMOOTHING_FRAMES
        )

        self.attention_score = 0

        # ====================================================
        # EMOTION
        # ====================================================

        self.emotion = "Unknown"
        self.emotion_confidence = 0

        self.last_emotion_time = 0

        # ====================================================
        # STATUS
        # ====================================================

        self.head_status = "Unknown"
        self.eye_status = "Unknown"
        self.mouth_status = "Unknown"
        self.attention_status = "Unknown"

        # ====================================================
        # FRAME COUNTER
        # ====================================================

        self.frame_count = 0

        # ====================================================
        # CSV
        # ====================================================

        self.last_log_time = time.time()

        # ====================================================
        # SESSION DATA
        # ====================================================

        self.session_attention = []
        self.session_confidence = []
        self.session_emotions = []

        self.session_start = time.time()

        # ====================================================
        # LOCK
        # ====================================================

        self.lock = threading.Lock()


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def recv(self, frame):

        # ====================================================
        # CONVERT FRAME
        # ====================================================

        img = frame.to_ndarray(
            format="bgr24"
        )

        # Mirror camera
        img = cv2.flip(
            img,
            1
        )

        # Resize to low-lag resolution
        img = cv2.resize(
            img,
            (
                CAMERA_WIDTH,
                CAMERA_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )

        height, width, _ = img.shape

        self.frame_count += 1


        # ====================================================
        # DEFAULT VALUES
        # ====================================================

        attention_score = self.attention_score

        dominant_emotion = self.emotion

        emotion_confidence = self.emotion_confidence

        eye_status = "Unknown"
        mouth_status = "Unknown"
        head_status = "Unknown"


        # ====================================================
        # MEDIAPIPE
        # ====================================================

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(
            rgb
        )


        # ====================================================
        # FACE DETECTED
        # ====================================================

        if results.multi_face_landmarks:

            face = results.multi_face_landmarks[0]

            landmarks = face.landmark

            raw_score = 100


            # =================================================
            # EYE DETECTION
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


            # =================================================
            # EYES
            # =================================================

            if ear < EAR_THRESHOLD:

                self.closed_frames += 1

                eye_status = "Eyes Closed"

                raw_score -= 20

            else:

                if self.closed_frames >= BLINK_FRAMES:

                    self.blink_count += 1

                self.closed_frames = 0

                eye_status = "Eyes Open"


            # =================================================
            # MOUTH
            # =================================================

            mouth_ratio = calculate_mouth_ratio(
                landmarks,
                width,
                height
            )


            if mouth_ratio > MOUTH_THRESHOLD:

                self.mouth_open_frames += 1

                mouth_status = "Mouth Open"

                raw_score -= 10

            else:

                if self.mouth_open_frames >= YAWN_FRAMES:

                    self.yawn_count += 1

                self.mouth_open_frames = 0

                mouth_status = "Mouth Closed"


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
            # ATTENTION SCORE
            # =================================================

            raw_score = max(
                0,
                min(
                    100,
                    raw_score
                )
            )


            self.score_history.append(
                raw_score
            )


            attention_score = (
                sum(
                    self.score_history
                )
                /
                len(
                    self.score_history
                )
            )


        # ====================================================
        # NO FACE
        # ====================================================

        else:

            head_status = "No Face"

            eye_status = "No Face"

            mouth_status = "No Face"

            attention_score = 0

            self.score_history.clear()


        # ====================================================
        # EMOTION DETECTION
        # ====================================================

        current_time = time.time()


        if (
            current_time -
            self.last_emotion_time
            >= EMOTION_INTERVAL_SECONDS
        ):

            self.last_emotion_time = current_time

            if results.multi_face_landmarks:

                try:

                    # ----------------------------------------
                    # FACE BOUNDING BOX
                    # ----------------------------------------

                    xs = [
                        int(point.x * width)
                        for point in landmarks
                    ]

                    ys = [
                        int(point.y * height)
                        for point in landmarks
                    ]

                    x1 = max(
                        0,
                        min(xs) - 20
                    )

                    y1 = max(
                        0,
                        min(ys) - 20
                    )

                    x2 = min(
                        width,
                        max(xs) + 20
                    )

                    y2 = min(
                        height,
                        max(ys) + 20
                    )

                    face_crop = img[
                        y1:y2,
                        x1:x2
                    ]


                    # ----------------------------------------
                    # VALID CROP
                    # ----------------------------------------

                    if (
                        face_crop is not None
                        and
                        face_crop.size > 0
                        and
                        face_crop.shape[0] > 30
                        and
                        face_crop.shape[1] > 30
                    ):

                        emotions = (
                            self.emotion_detector
                            .detect_emotions(
                                face_crop
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
                                ] * 100
                            )


                except Exception:

                    pass


        # ====================================================
        # PERFORMANCE STATUS
        # ====================================================

        if attention_score >= 80:

            attention_status = "Attentive"

        elif attention_score >= 50:

            attention_status = "Moderate"

        else:

            attention_status = "Distracted"


        # ====================================================
        # SESSION DATA
        # ====================================================

        self.session_attention.append(
            attention_score
        )

        self.session_confidence.append(
            emotion_confidence
        )


        if dominant_emotion != "Unknown":

            self.session_emotions.append(
                dominant_emotion
            )


        # ====================================================
        # CSV LOGGING
        # ====================================================

        current_time = time.time()


        if (
            current_time -
            self.last_log_time
            >= LOG_INTERVAL
        ):

            try:

                file_exists = os.path.exists(
                    CSV_FILE
                )

                file_empty = (
                    not file_exists
                    or
                    os.path.getsize(
                        CSV_FILE
                    ) == 0
                )


                with open(
                    CSV_FILE,
                    "a",
                    newline="",
                    encoding="utf-8"
                ) as file:

                    writer = csv.writer(
                        file
                    )


                    if file_empty:

                        writer.writerow([
                            "Time",
                            "Attention",
                            "Emotion",
                            "Confidence",
                            "Blinks",
                            "Yawns",
                            "Head",
                            "Eyes",
                            "Mouth"
                        ])


                    writer.writerow([
                        time.strftime(
                            "%H:%M:%S"
                        ),

                        round(
                            attention_score
                        ),

                        dominant_emotion,

                        round(
                            emotion_confidence,
                            2
                        ),

                        self.blink_count,

                        self.yawn_count,

                        head_status,

                        eye_status,

                        mouth_status
                    ])


            except Exception:

                pass


            self.last_log_time = current_time


        # ====================================================
        # UPDATE SHARED DATA
        # ====================================================

        with self.lock:

            self.attention_score = (
                attention_score
            )

            self.emotion = (
                dominant_emotion
            )

            self.emotion_confidence = (
                emotion_confidence
            )

            self.head_status = (
                head_status
            )

            self.eye_status = (
                eye_status
            )

            self.mouth_status = (
                mouth_status
            )

            # IMPORTANT:
            # Performance is updated here every frame
            self.attention_status = (
                attention_status
            )


        # ====================================================
        # CAMERA INFORMATION BOX
        # ====================================================

        box_width = 300
        box_height = 235

        overlay = img.copy()


        cv2.rectangle(
            overlay,
            (10, 10),
            (
                box_width,
                box_height
            ),
            (0, 0, 0),
            -1
        )


        img = cv2.addWeighted(
            overlay,
            0.75,
            img,
            0.25,
            0
        )


        # ====================================================
        # TEXT
        # ====================================================

        cv2.putText(
            img,
            f"Attention: {round(attention_score)}%",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            img,
            f"Performance: {attention_status}",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )


        cv2.putText(
            img,
            f"Emotion: {dominant_emotion}",
            (20, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )


        cv2.putText(
            img,
            f"Confidence: {round(emotion_confidence)}%",
            (20, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )


        cv2.putText(
            img,
            f"Head: {head_status}",
            (20, 158),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )


        cv2.putText(
            img,
            f"Eyes: {eye_status}",
            (20, 188),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )


        cv2.putText(
            img,
            f"Blinks: {self.blink_count}",
            (20, 218),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2
        )


        cv2.putText(
            img,
            f"Yawns: {self.yawn_count}",
            (150, 218),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            2
        )


        # ====================================================
        # RETURN CAMERA FRAME
        # ====================================================

        return frame.from_ndarray(
            img,
            format="bgr24"
        )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🧠 Live Facial Emotion & Attentiveness Monitor"
)

st.caption(
    "Real-time AI-based facial analysis and attention monitoring"
)

st.divider()


# ============================================================
# CAMERA
# ============================================================

st.subheader(
    "📹 Live AI Camera"
)


# ============================================================
# WEBRTC CONFIGURATION
# ============================================================

RTC_CONFIGURATION = {
    "iceServers": [
        {
            "urls": [
                "stun:stun.l.google.com:19302"
            ]
        }
    ]
}


ctx = webrtc_streamer(

    key="stable-ai-facial-monitor",

    video_processor_factory=VideoProcessor,

    rtc_configuration=RTC_CONFIGURATION,

    media_stream_constraints={
        "video": {
            "width": {
                "ideal": CAMERA_WIDTH
            },

            "height": {
                "ideal": CAMERA_HEIGHT
            },

            "frameRate": {
                "ideal": CAMERA_FPS,
                "max": CAMERA_FPS
            }
        },

        "audio": False
    },

    async_processing=True
)


# ============================================================
# FIX: AUTO REFRESH DASHBOARD
# ============================================================
#
# WebRTC processes frames separately from Streamlit.
# Therefore Streamlit needs to rerun periodically to display
# the latest Performance / Attention values.
#

if ctx.state.playing:

    st_autorefresh(
        interval=1000,
        key="dashboard_refresh"
    )


# ============================================================
# LIVE DASHBOARD
# ============================================================

if ctx.video_processor:

    processor = ctx.video_processor


    # ========================================================
    # READ VALUES
    # ========================================================

    with processor.lock:

        attention = (
            processor.attention_score
        )

        emotion = (
            processor.emotion
        )

        confidence = (
            processor.emotion_confidence
        )

        blinks = (
            processor.blink_count
        )

        yawns = (
            processor.yawn_count
        )

        head = (
            processor.head_status
        )

        eyes = (
            processor.eye_status
        )

        mouth = (
            processor.mouth_status
        )

        performance = (
            processor.attention_status
        )


    # ========================================================
    # LIVE PERFORMANCE
    # ========================================================

    st.divider()

    st.subheader(
        "⚡ Live Performance"
    )


    p1, p2, p3 = st.columns(3)


    with p1:

        st.metric(
            "🎯 Attention",
            f"{round(attention)}%"
        )


    with p2:

        st.metric(
            "📊 Performance",
            performance
        )


    with p3:

        st.metric(
            "🎯 Emotion Confidence",
            f"{round(confidence)}%"
        )


    # ========================================================
    # LIVE METRICS
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Live Metrics"
    )


    c1, c2, c3, c4, c5 = st.columns(5)


    with c1:

        st.metric(
            "🎯 Attention",
            f"{round(attention)}%"
        )


    with c2:

        st.metric(
            "😊 Emotion",
            emotion.capitalize()
        )


    with c3:

        st.metric(
            "🎯 Confidence",
            f"{round(confidence)}%"
        )


    with c4:

        st.metric(
            "👁 Blinks",
            blinks
        )


    with c5:

        st.metric(
            "🥱 Yawns",
            yawns
        )


    # ========================================================
    # CURRENT ANALYSIS
    # ========================================================

    st.divider()


    left, right = st.columns(2)


    with left:

        st.subheader(
            "📊 Current Status"
        )

        st.write(
            f"**Attention Score:** "
            f"{round(attention)}%"
        )

        st.write(
            f"**Performance:** "
            f"{performance}"
        )

        st.write(
            f"**Emotion:** "
            f"{emotion.capitalize()}"
        )

        st.write(
            f"**Emotion Confidence:** "
            f"{round(confidence)}%"
        )


    with right:

        st.subheader(
            "👤 Facial Analysis"
        )

        st.write(
            f"**Head Direction:** {head}"
        )

        st.write(
            f"**Eyes:** {eyes}"
        )

        st.write(
            f"**Mouth:** {mouth}"
        )

        st.write(
            f"**Blink Count:** {blinks}"
        )

        st.write(
            f"**Yawn Count:** {yawns}"
        )


else:

    st.info(
        "📷 Click START to begin AI monitoring."
    )


# ============================================================
# MONITORING HISTORY
# ============================================================

st.divider()

st.header(
    "📈 Monitoring History"
)


def load_monitoring_history():

    if not os.path.exists(
        CSV_FILE
    ):

        return pd.DataFrame()


    try:

        rows = []


        with open(
            CSV_FILE,
            "r",
            newline="",
            encoding="utf-8-sig",
            errors="ignore"
        ) as file:

            reader = csv.reader(
                file
            )


            for row in reader:

                if row:

                    rows.append(row)


        if not rows:

            return pd.DataFrame()


        columns = [
            "Time",
            "Attention",
            "Emotion",
            "Confidence",
            "Blinks",
            "Yawns",
            "Head",
            "Eyes",
            "Mouth"
        ]


        # ====================================================
        # FIND HEADER
        # ====================================================

        header_index = None


        for i, row in enumerate(rows):

            cleaned = [
                str(x).strip().lower()
                for x in row
            ]


            if (
                "time" in cleaned
                and
                "attention" in cleaned
            ):

                header_index = i

                break


        if header_index is None:

            return pd.DataFrame()


        data_rows = rows[
            header_index + 1:
        ]


        clean_rows = []


        for row in data_rows:

            if not any(
                str(x).strip()
                for x in row
            ):

                continue


            if len(row) < 9:

                row += [
                    ""
                ] * (
                    9 - len(row)
                )


            elif len(row) > 9:

                row = row[:9]


            clean_rows.append(
                row
            )


        if not clean_rows:

            return pd.DataFrame()


        df = pd.DataFrame(
            clean_rows,
            columns=columns
        )


        # ====================================================
        # NUMERIC
        # ====================================================

        numeric_columns = [
            "Attention",
            "Confidence",
            "Blinks",
            "Yawns"
        ]


        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )


        df = df.dropna(
            subset=[
                "Attention"
            ]
        )


        df["Attention"] = (
            df["Attention"]
            .fillna(0)
            .round()
            .astype(int)
        )


        df["Confidence"] = (
            df["Confidence"]
            .fillna(0)
            .round(2)
        )


        df["Blinks"] = (
            df["Blinks"]
            .fillna(0)
            .round()
            .astype(int)
        )


        df["Yawns"] = (
            df["Yawns"]
            .fillna(0)
            .round()
            .astype(int)
        )


        # ====================================================
        # TEXT
        # ====================================================

        for column in [
            "Time",
            "Emotion",
            "Head",
            "Eyes",
            "Mouth"
        ]:

            df[column] = (
                df[column]
                .fillna("Unknown")
                .astype(str)
            )


        return df


    except Exception:

        return pd.DataFrame()


# ============================================================
# LOAD HISTORY
# ============================================================

df = load_monitoring_history()


if not df.empty:

    # ========================================================
    # ATTENTION GRAPH
    # ========================================================

    st.subheader(
        "📊 Attention Analysis"
    )


    attention_graph = df[
        [
            "Time",
            "Attention"
        ]
    ].copy()


    attention_graph["Time"] = (
        attention_graph["Time"]
        .astype(str)
    )


    attention_graph = (
        attention_graph
        .set_index("Time")
    )


    st.line_chart(
        attention_graph,
        height=350
    )


    # ========================================================
    # CONFIDENCE GRAPH
    # ========================================================

    st.subheader(
        "🎯 Emotion Confidence Analysis"
    )


    confidence_graph = df[
        [
            "Time",
            "Confidence"
        ]
    ].copy()


    confidence_graph["Time"] = (
        confidence_graph["Time"]
        .astype(str)
    )


    confidence_graph = (
        confidence_graph
        .set_index("Time")
    )


    st.line_chart(
        confidence_graph,
        height=300
    )


    # ========================================================
    # HISTORY TABLE
    # ========================================================

    st.subheader(
        "📋 Detailed Monitoring History"
    )


    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # HISTORY SUMMARY
    # ========================================================

    st.subheader(
        "📌 History Summary"
    )


    h1, h2, h3, h4 = st.columns(4)


    with h1:

        st.metric(
            "Average Attention",
            f"{round(df['Attention'].mean())}%"
        )


    with h2:

        st.metric(
            "Highest Attention",
            f"{round(df['Attention'].max())}%"
        )


    with h3:

        st.metric(
            "Lowest Attention",
            f"{round(df['Attention'].min())}%"
        )


    with h4:

        st.metric(
            "Average Confidence",
            f"{round(df['Confidence'].mean())}%"
        )


else:

    st.info(
        "No valid monitoring history yet. "
        "Start the camera to generate new data."
    )


# ============================================================
# SESSION SUMMARY
# ============================================================

st.divider()

st.header(
    "🧠 Session Summary"
)


if ctx.video_processor:

    processor = ctx.video_processor


    with processor.lock:

        attention_values = list(
            processor.session_attention
        )

        confidence_values = list(
            processor.session_confidence
        )

        session_emotions = list(
            processor.session_emotions
        )

        session_blinks = (
            processor.blink_count
        )

        session_yawns = (
            processor.yawn_count
        )


    if attention_values:

        # ====================================================
        # CALCULATIONS
        # ====================================================

        avg_attention = (
            sum(attention_values)
            /
            len(attention_values)
        )


        highest_attention = max(
            attention_values
        )


        lowest_attention = min(
            attention_values
        )


        if confidence_values:

            avg_confidence = (
                sum(confidence_values)
                /
                len(confidence_values)
            )

        else:

            avg_confidence = 0


        # ====================================================
        # METRICS
        # ====================================================

        s1, s2, s3, s4 = st.columns(4)


        with s1:

            st.metric(
                "Average Attention",
                f"{round(avg_attention)}%"
            )


        with s2:

            st.metric(
                "Highest Attention",
                f"{round(highest_attention)}%"
            )


        with s3:

            st.metric(
                "Lowest Attention",
                f"{round(lowest_attention)}%"
            )


        with s4:

            st.metric(
                "Average Confidence",
                f"{round(avg_confidence)}%"
            )


        st.write(
            f"**Total Blinks:** "
            f"{session_blinks}"
        )


        st.write(
            f"**Total Yawns:** "
            f"{session_yawns}"
        )


        # ====================================================
        # MOST COMMON EMOTION
        # ====================================================

        if session_emotions:

            emotion_series = pd.Series(
                session_emotions
            )


            most_common_emotion = (
                emotion_series
                .value_counts()
                .index[0]
            )


            st.write(
                f"**Most Detected Emotion:** "
                f"{most_common_emotion.capitalize()}"
            )


    else:

        st.info(
            "Session summary will appear "
            "after monitoring starts."
        )