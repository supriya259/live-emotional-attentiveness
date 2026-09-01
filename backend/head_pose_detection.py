import cv2
import mediapipe as mp
import numpy as np

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh

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

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        h, w, _ = frame.shape

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                # Landmark indices
                nose = face_landmarks.landmark[1]
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                chin = face_landmarks.landmark[199]
                left_mouth = face_landmarks.landmark[61]
                right_mouth = face_landmarks.landmark[291]

                image_points = np.array([
                    (nose.x*w, nose.y*h),
                    (chin.x*w, chin.y*h),
                    (left_eye.x*w, left_eye.y*h),
                    (right_eye.x*w, right_eye.y*h),
                    (left_mouth.x*w, left_mouth.y*h),
                    (right_mouth.x*w, right_mouth.y*h)
                ], dtype="double")

                # 3D face model points
                model_points = np.array([
                    (0.0, 0.0, 0.0),
                    (0.0, -63.6, -12.5),
                    (-43.3, 32.7, -26),
                    (43.3, 32.7, -26),
                    (-28.9, -28.9, -24.1),
                    (28.9, -28.9, -24.1)
                ])

                focal_length = w
                center = (w/2, h/2)

                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype="double")

                dist_coeffs = np.zeros((4,1))

                success, rotation_vector, translation_vector = cv2.solvePnP(
                    model_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs
                )

                rmat, _ = cv2.Rodrigues(rotation_vector)

                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360

                if y < -10:
                    text = "Looking Left"

                elif y > 10:
                    text = "Looking Right"

                elif x < -10:
                    text = "Looking Down"

                elif x > 10:
                    text = "Looking Up"

                else:
                    text = "Looking Forward"

                cv2.putText(
                    frame,
                    text,
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

                # Draw landmark points
                for point in image_points:
                    cv2.circle(
                        frame,
                        (int(point[0]), int(point[1])),
                        4,
                        (0,0,255),
                        -1
                    )

        cv2.imshow("Day 7 - Head Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()