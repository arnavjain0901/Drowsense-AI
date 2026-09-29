# detector.py

import cv2
import mediapipe as mp
import numpy as np
import math


class DrowsinessDetector:
    """
    Handles MediaPipe Face Mesh detection and calculates:

    - Eye Aspect Ratio (EAR)
    - Mouth Aspect Ratio (MAR)
    - Head pose
    - Facial landmarks

    If multiple faces are detected, the largest face is selected.
    """

    # ---------------------------------------------------------
    # MediaPipe facial landmark indices
    # ---------------------------------------------------------

    # Left eye
    LEFT_EYE = [33, 160, 158, 133, 153, 144]

    # Right eye
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    # Mouth
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291

    # Head-pose landmarks
    NOSE = 1
    CHIN = 152
    LEFT_EYE_CORNER = 33
    RIGHT_EYE_CORNER = 263
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    def __init__(self, max_faces=5):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    # ---------------------------------------------------------
    # Distance between two points
    # ---------------------------------------------------------

    @staticmethod
    def distance(p1, p2):

        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    # ---------------------------------------------------------
    # Calculate Eye Aspect Ratio
    # ---------------------------------------------------------

    def calculate_ear(self, landmarks, eye_indices):
        """
        EAR measures how open the eye is.

        Higher EAR  = eye more open
        Lower EAR   = eye more closed
        """

        points = [
            (landmarks[i][0], landmarks[i][1])
            for i in eye_indices
        ]

        p1, p2, p3, p4, p5, p6 = points

        vertical_1 = self.distance(p2, p6)
        vertical_2 = self.distance(p3, p5)

        horizontal = self.distance(p1, p4)

        if horizontal == 0:
            return 0.0

        ear = (
            vertical_1 + vertical_2
        ) / (2.0 * horizontal)

        return ear

    # ---------------------------------------------------------
    # Calculate Mouth Aspect Ratio
    # ---------------------------------------------------------

    def calculate_mar(self, landmarks):
        """
        MAR measures mouth opening.

        Higher MAR = mouth more open
        """

        top = (
            landmarks[self.MOUTH_TOP][0],
            landmarks[self.MOUTH_TOP][1]
        )

        bottom = (
            landmarks[self.MOUTH_BOTTOM][0],
            landmarks[self.MOUTH_BOTTOM][1]
        )

        left = (
            landmarks[self.MOUTH_LEFT][0],
            landmarks[self.MOUTH_LEFT][1]
        )

        right = (
            landmarks[self.MOUTH_RIGHT][0],
            landmarks[self.MOUTH_RIGHT][1]
        )

        vertical = self.distance(top, bottom)
        horizontal = self.distance(left, right)

        if horizontal == 0:
            return 0.0

        return vertical / horizontal

    # ---------------------------------------------------------
    # Head pose estimation
    # ---------------------------------------------------------

    def get_head_pose(
        self,
        landmarks,
        frame_width,
        frame_height
    ):
        """
        Estimate head pitch, yaw and roll using solvePnP.
        """

        image_points = np.array([
            landmarks[self.NOSE],
            landmarks[self.CHIN],
            landmarks[self.LEFT_EYE_CORNER],
            landmarks[self.RIGHT_EYE_CORNER],
            landmarks[self.LEFT_MOUTH],
            landmarks[self.RIGHT_MOUTH]
        ], dtype=np.float64)

        # Approximate 3D face model.
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)

        focal_length = frame_width

        center = (
            frame_width / 2,
            frame_height / 2
        )

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        distortion = np.zeros((4, 1))

        try:

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return 0.0, 0.0, 0.0

            rotation_matrix, _ = cv2.Rodrigues(
                rotation_vector
            )

            pose_matrix = np.hstack(
                (rotation_matrix, translation_vector)
            )

            _, _, _, _, _, _, euler_angles = (
                cv2.decomposeProjectionMatrix(
                    pose_matrix
                )
            )

            pitch = float(euler_angles[0])
            yaw = float(euler_angles[1])
            roll = float(euler_angles[2])

            return pitch, yaw, roll

        except Exception:

            return 0.0, 0.0, 0.0

    # ---------------------------------------------------------
    # Select largest face
    # ---------------------------------------------------------

    def select_largest_face(
        self,
        face_landmarks,
        width,
        height
    ):
        """
        If multiple faces are present, select the largest one.
        """

        largest_face = None
        largest_area = 0

        for face in face_landmarks:

            points = np.array([
                [
                    lm.x * width,
                    lm.y * height
                ]
                for lm in face.landmark
            ])

            x_min = np.min(points[:, 0])
            x_max = np.max(points[:, 0])

            y_min = np.min(points[:, 1])
            y_max = np.max(points[:, 1])

            area = (
                max(0, x_max - x_min)
                *
                max(0, y_max - y_min)
            )

            if area > largest_area:

                largest_area = area
                largest_face = face

        return largest_face

    # ---------------------------------------------------------
    # Process webcam frame
    # ---------------------------------------------------------

    def process(self, frame):
        """
        Process one webcam frame.

        Returns a dictionary containing:

        face_detected
        EAR
        MAR
        pitch
        yaw
        roll
        landmarks
        """

        height, width = frame.shape[:2]

        # OpenCV uses BGR.
        # MediaPipe expects RGB.
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb)

        # No face detected.
        if not results.multi_face_landmarks:

            return {
                "face_detected": False,
                "ear": 0.0,
                "mar": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "landmarks": None
            }

        # Select largest face.
        face = self.select_largest_face(
            results.multi_face_landmarks,
            width,
            height
        )

        if face is None:

            return {
                "face_detected": False,
                "ear": 0.0,
                "mar": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "landmarks": None
            }

        # Convert normalized MediaPipe coordinates
        # into actual pixel coordinates.
        landmarks = []

        for lm in face.landmark:

            landmarks.append(
                (
                    int(lm.x * width),
                    int(lm.y * height)
                )
            )

        # Calculate left and right EAR.
        left_ear = self.calculate_ear(
            landmarks,
            self.LEFT_EYE
        )

        right_ear = self.calculate_ear(
            landmarks,
            self.RIGHT_EYE
        )

        # Average both eyes.
        ear = (
            left_ear +
            right_ear
        ) / 2.0

        # Calculate mouth opening.
        mar = self.calculate_mar(
            landmarks
        )

        # Calculate head pose.
        pitch, yaw, roll = self.get_head_pose(
            landmarks,
            width,
            height
        )

        return {
            "face_detected": True,
            "ear": ear,
            "mar": mar,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "landmarks": landmarks
        }

    # ---------------------------------------------------------
    # Draw important landmarks
    # ---------------------------------------------------------

    def draw_landmarks(self, frame, data):

        if not data["face_detected"]:
            return frame

        landmarks = data["landmarks"]

        # Eyes
        for index in (
            self.LEFT_EYE +
            self.RIGHT_EYE
        ):

            x, y = landmarks[index]

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

        # Mouth
        for index in [
            self.MOUTH_TOP,
            self.MOUTH_BOTTOM,
            self.MOUTH_LEFT,
            self.MOUTH_RIGHT
        ]:

            x, y = landmarks[index]

            cv2.circle(
                frame,
                (x, y),
                3,
                (255, 0, 0),
                -1
            )

        return frame

    # ---------------------------------------------------------
    # Close MediaPipe
    # ---------------------------------------------------------

    def close(self):

        self.face_mesh.close()