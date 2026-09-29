# main.py

import cv2
import time
from collections import deque

from detector import DrowsinessDetector
from calibration import EARCalibrator
from alerts import AlertSystem


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

# Calibration
CALIBRATION_TIME = 5.0
EAR_THRESHOLD_FACTOR = 0.70

# Eye closure
# At ~30 FPS, 15 frames is roughly 0.5 seconds.
EAR_CONSECUTIVE_FRAMES = 15

# Yawning
MAR_THRESHOLD = 0.65
YAWN_CONSECUTIVE_FRAMES = 12

# Nodding
NOD_CONSECUTIVE_FRAMES = 8

# Displayed score at which status becomes WARNING.
DROWSINESS_SCORE_THRESHOLD = 60


# ============================================================
# FPS COUNTER
# ============================================================

class FPSCounter:

    def __init__(self):

        self.previous_time = time.time()
        self.fps = 0.0

    def update(self):

        current_time = time.time()

        elapsed = current_time - self.previous_time

        if elapsed > 0:

            instant_fps = 1.0 / elapsed

            # Smooth the FPS value.
            if self.fps == 0:
                self.fps = instant_fps
            else:
                self.fps = (
                    0.9 * self.fps +
                    0.1 * instant_fps
                )

        self.previous_time = current_time

        return self.fps


# ============================================================
# CALIBRATION DISPLAY
# ============================================================

def calibration_display(
    frame,
    remaining,
    samples
):
    """
    Display instructions during the 5-second calibration.
    """

    # Dark banner.
    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 105),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "CALIBRATING - LOOK AT CAMERA NORMALLY",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Remaining: {remaining:.1f}s",
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Samples: {samples}",
        (250, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return frame


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("AI DRIVER DROWSINESS DETECTION SYSTEM")
    print("=" * 60)

    print("\nOpening webcam...")

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():

        print("\nERROR: Could not open webcam.")
        print("Try changing CAMERA_INDEX from 0 to 1.")

        return

    # Request 1280x720.
    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    # --------------------------------------------------------
    # CREATE MODULES
    # --------------------------------------------------------

    detector = DrowsinessDetector(
        max_faces=5
    )

    alerts = AlertSystem(
        log_file="drowsiness_log.csv"
    )

    fps_counter = FPSCounter()

    # ========================================================
    # CALIBRATION
    # ========================================================

    print("\nCalibration starting...")
    print(
        "Look normally at the camera for "
        f"{CALIBRATION_TIME} seconds."
    )

    calibrator = EARCalibrator(
        duration=CALIBRATION_TIME,
        threshold_factor=EAR_THRESHOLD_FACTOR
    )

    # Get camera dimensions.
    ret, test_frame = camera.read()

    if not ret:

        print("\nERROR: Camera frame could not be read.")

        camera.release()
        detector.close()
        alerts.close()

        return

    frame_height, frame_width = test_frame.shape[:2]

    # Perform calibration.
    ear_threshold = calibrator.calibrate(
        detector,
        camera,
        frame_width,
        frame_height,
        calibration_display
    )

    # Calibration failed.
    if ear_threshold is None:

        print("\nCalibration failed.")

        print(
            "Make sure your face is visible and "
            "there is enough light."
        )

        camera.release()
        detector.close()
        alerts.close()

        cv2.destroyAllWindows()

        return

    print("\nCalibration successful!")

    print(
        f"Personalized EAR threshold: "
        f"{ear_threshold:.3f}"
    )

    # ========================================================
    # STATE VARIABLES
    # ========================================================

    # Number of frames with eyes below threshold.
    eye_closed_frames = 0

    # Number of frames with mouth open.
    yawn_frames = 0

    # Number of frames associated with downward head movement.
    nod_frames = 0

    # Head-pitch history.
    pitch_history = deque(
        maxlen=20
    )

    # Yawn logging cooldown.
    last_yawn_alert = 0

    YAWN_ALERT_COOLDOWN = 5

    print("\n")
    print("=" * 60)
    print("SYSTEM ACTIVE")
    print("=" * 60)
    print("Q = Quit")
    print("R = Recalibrate")
    print("=" * 60)

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        ret, frame = camera.read()

        if not ret:

            print("Unable to read camera frame.")

            break

        # Mirror the camera.
        frame = cv2.flip(
            frame,
            1
        )

        # ----------------------------------------------------
        # PROCESS FRAME
        # ----------------------------------------------------

        data = detector.process(
            frame
        )

        fps = fps_counter.update()

        face_detected = data["face_detected"]

        ear = data["ear"]

        mar = data["mar"]

        pitch = data["pitch"]

        yaw = data["yaw"]

        roll = data["roll"]

        # ----------------------------------------------------
        # NO FACE
        # ----------------------------------------------------

        if not face_detected:

            eye_closed_frames = 0
            yawn_frames = 0
            nod_frames = 0

            # Slowly decrease displayed score.
            drowsiness_score = 0

            alerts.stop_alarm()

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 165, 255),
                2
            )

        # ----------------------------------------------------
        # FACE DETECTED
        # ----------------------------------------------------

        else:

            # Draw eye and mouth landmarks.
            detector.draw_landmarks(
                frame,
                data
            )

            # =================================================
            # EYE CLOSURE
            # =================================================

            if ear < ear_threshold:

                eye_closed_frames += 1

            else:

                # Slowly recover instead of instantly
                # resetting on one noisy frame.
                eye_closed_frames = max(
                    0,
                    eye_closed_frames - 2
                )

            # =================================================
            # YAWNING
            # =================================================

            if mar > MAR_THRESHOLD:

                yawn_frames += 1

            else:

                yawn_frames = max(
                    0,
                    yawn_frames - 2
                )

            # =================================================
            # HEAD MOVEMENT
            # =================================================

            pitch_history.append(
                pitch
            )

            if len(pitch_history) >= 2:

                pitch_change = (
                    pitch_history[-1] -
                    pitch_history[-2]
                )

                # Positive pitch movement is treated
                # as possible downward movement.
                if pitch_change > 1.5:

                    nod_frames += 1

                else:

                    nod_frames = max(
                        0,
                        nod_frames - 1
                    )

            # =================================================
            # CALCULATE DROWSINESS SCORE
            # =================================================

            # Eye component = 0 to 60.
            if (
                eye_closed_frames >=
                EAR_CONSECUTIVE_FRAMES
            ):

                eye_score = 60

            else:

                eye_score = int(
                    60 *
                    eye_closed_frames /
                    EAR_CONSECUTIVE_FRAMES
                )

            # Yawn component = 0 to 20.
            if (
                yawn_frames >=
                YAWN_CONSECUTIVE_FRAMES
            ):

                yawn_score = 20

            else:

                yawn_score = int(
                    20 *
                    yawn_frames /
                    YAWN_CONSECUTIVE_FRAMES
                )

            # Nodding component = 0 to 20.
            if (
                nod_frames >=
                NOD_CONSECUTIVE_FRAMES
            ):

                nod_score = 20

            else:

                nod_score = int(
                    20 *
                    nod_frames /
                    NOD_CONSECUTIVE_FRAMES
                )

            # Total = 0 to 100.
            drowsiness_score = min(
                100,
                eye_score +
                yawn_score +
                nod_score
            )

            # =================================================
            # DROWSINESS DECISION
            # =================================================

            # Primary trigger:
            # Eyes remain closed for enough frames.
            eye_drowsiness = (
                eye_closed_frames >=
                EAR_CONSECUTIVE_FRAMES
            )

            # Secondary trigger:
            # Long yawn + nodding.
            secondary_drowsiness = (
                yawn_frames >=
                YAWN_CONSECUTIVE_FRAMES
                and
                nod_frames >=
                NOD_CONSECUTIVE_FRAMES
            )

            drowsiness_detected = (
                eye_drowsiness
                or
                secondary_drowsiness
            )

            # =================================================
            # ALERT
            # =================================================

            if drowsiness_detected:

                alerts.start_alarm()

                frame = alerts.draw_warning(
                    frame,
                    "DROWSINESS ALERT!"
                )

            else:

                alerts.stop_alarm()

            # =================================================
            # YAWN LOGGING
            # =================================================

            current_time = time.time()

            if (
                yawn_frames >=
                YAWN_CONSECUTIVE_FRAMES
                and
                current_time -
                last_yawn_alert >
                YAWN_ALERT_COOLDOWN
            ):

                alerts.log_event(
                    "YAWNING"
                )

                last_yawn_alert = current_time

        # ====================================================
        # INFORMATION PANEL
        # ====================================================

        panel_height = 190

        panel_width = 450

        cv2.rectangle(
            frame,
            (
                0,
                frame.shape[0] - panel_height
            ),
            (
                panel_width,
                frame.shape[0]
            ),
            (20, 20, 20),
            -1
        )

        # EAR
        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (
                15,
                frame.shape[0] - 155
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # EAR threshold
        cv2.putText(
            frame,
            f"EAR Threshold: {ear_threshold:.3f}",
            (
                15,
                frame.shape[0] - 125
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1
        )

        # MAR
        cv2.putText(
            frame,
            f"MAR: {mar:.3f}",
            (
                15,
                frame.shape[0] - 95
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (
                250,
                frame.shape[0] - 155
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # Head pitch
        cv2.putText(
            frame,
            f"Pitch: {pitch:.1f}",
            (
                250,
                frame.shape[0] - 125
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1
        )

        # Drowsiness score
        cv2.putText(
            frame,
            f"Drowsiness: {drowsiness_score}/100",
            (
                15,
                frame.shape[0] - 55
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        # Status
        if drowsiness_score >= DROWSINESS_SCORE_THRESHOLD:

            status = "WARNING"

            status_color = (
                0,
                0,
                255
            )

        else:

            status = "NORMAL"

            status_color = (
                0,
                255,
                0
            )

        cv2.putText(
            frame,
            status,
            (
                250,
                frame.shape[0] - 55
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            status_color,
            2
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "Driver Drowsiness Detection",
            frame
        )

        # Wait 1 ms for keyboard input.
        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if key == ord("q"):

            break

        # ----------------------------------------------------
        # RECALIBRATE
        # ----------------------------------------------------

        elif key == ord("r"):

            print("\nRecalibrating...")

            pitch_history.clear()

            eye_closed_frames = 0
            yawn_frames = 0
            nod_frames = 0

            alerts.stop_alarm()

            ear_threshold = calibrator.calibrate(
                detector,
                camera,
                frame_width,
                frame_height,
                calibration_display
            )

            if ear_threshold is None:

                print(
                    "Recalibration failed."
                )

                break

            print(
                f"New EAR threshold: "
                f"{ear_threshold:.3f}"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    print("\nShutting down...")

    alerts.stop_alarm()

    camera.release()

    detector.close()

    alerts.close()

    cv2.destroyAllWindows()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()