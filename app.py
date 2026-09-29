from flask import Flask, Response, render_template, jsonify
import cv2
import time
import threading
import webbrowser
from collections import deque
import numpy as np

from detector import DrowsinessDetector
from alerts import AlertSystem


app = Flask(__name__)


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 0

CALIBRATION_DURATION = 5.0
CALIBRATION_FACTOR = 0.70

MIN_EAR_THRESHOLD = 0.15
MAX_EAR_THRESHOLD = 0.32

EAR_CONSECUTIVE_FRAMES = 15

MAR_THRESHOLD = 0.65
YAWN_CONSECUTIVE_FRAMES = 12

NOD_CONSECUTIVE_FRAMES = 8
NOD_PITCH_CHANGE = 1.5

DROWSINESS_SCORE_THRESHOLD = 60


# ============================================================
# DROWSINESS WEB SYSTEM
# ============================================================

class DrowsinessWebSystem:

    def __init__(self):

        self.camera = None

        self.detector = DrowsinessDetector(max_faces=5)
        self.alerts = AlertSystem()

        self.running = False

        self.thread = None

        self.lock = threading.Lock()

        self.latest_frame = None

        self.recalibrate_requested = True

        # Detection values
        self.ear = 0.0
        self.mar = 0.0

        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0

        self.ear_threshold = None

        self.drowsiness_score = 0

        self.status = "STARTING"

        self.face_detected = False

        self.fps = 0.0

        self.calibration_remaining = 0

        # Counters
        self.eye_closed_frames = 0
        self.yawn_frames = 0
        self.nod_frames = 0

        self.pitch_history = deque(maxlen=20)

        self.last_yawn_log = 0


    # ========================================================
    # START
    # ========================================================

    def start(self):

        self.camera = cv2.VideoCapture(CAMERA_INDEX)

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

        if not self.camera.isOpened():

            self.status = "CAMERA ERROR"

            print("ERROR: Could not open camera.")

            return False

        self.running = True

        self.thread = threading.Thread(
            target=self.process_camera,
            daemon=True
        )

        self.thread.start()

        return True


    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        self.running = False

        try:
            self.alerts.stop_alarm()
        except Exception:
            pass

        if self.camera is not None:

            self.camera.release()

        try:
            self.detector.close()
        except Exception:
            pass

        try:
            self.alerts.close()
        except Exception:
            pass


    # ========================================================
    # REQUEST RECALIBRATION
    # ========================================================

    def request_recalibration(self):

        self.recalibrate_requested = True


    # ========================================================
    # CALIBRATION
    # ========================================================

    def calibrate(self):

        print("\nStarting EAR calibration...")
        print("Look normally at the camera for 5 seconds.")

        self.status = "CALIBRATING"

        self.calibration_remaining = CALIBRATION_DURATION

        values = []

        start_time = time.time()

        while (
            self.running
            and time.time() - start_time < CALIBRATION_DURATION
        ):

            success, frame = self.camera.read()

            if not success:
                continue

            frame = cv2.flip(frame, 1)

            data = self.detector.process(frame)

            if data["face_detected"]:

                ear = data["ear"]

                if 0.05 < ear < 0.60:

                    values.append(ear)

                self.detector.draw_landmarks(
                    frame,
                    data
                )

            remaining = (
                CALIBRATION_DURATION
                - (time.time() - start_time)
            )

            self.calibration_remaining = max(
                0,
                round(remaining, 1)
            )

            # Calibration overlay

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (frame.shape[1], 110),
                (0, 0, 0),
                -1
            )

            frame = cv2.addWeighted(
                overlay,
                0.65,
                frame,
                0.35,
                0
            )

            cv2.putText(
                frame,
                "CALIBRATING DROWSINESS DETECTOR",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Look normally at the camera",
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Time remaining: {remaining:.1f}s",
                (25, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            self.publish_frame(frame)

        # ----------------------------------------------------
        # Calculate threshold
        # ----------------------------------------------------

        if len(values) < 10:

            print("Calibration failed: not enough EAR samples.")

            self.status = "CALIBRATION FAILED"

            self.ear_threshold = None

            return False

        median_ear = float(
            np.median(values)
        )

        threshold = median_ear * CALIBRATION_FACTOR

        threshold = max(
            MIN_EAR_THRESHOLD,
            min(
                MAX_EAR_THRESHOLD,
                threshold
            )
        )

        self.ear_threshold = threshold

        self.recalibrate_requested = False

        self.status = "NORMAL"

        print(
            f"Calibration complete."
            f" Median EAR: {median_ear:.3f}"
            f" | Threshold: {threshold:.3f}"
        )

        return True


    # ========================================================
    # MAIN CAMERA PROCESSING LOOP
    # ========================================================

    def process_camera(self):

        previous_time = time.time()

        while self.running:

            # ------------------------------------------------
            # Recalibration
            # ------------------------------------------------

            if (
                self.recalibrate_requested
                or self.ear_threshold is None
            ):

                success = self.calibrate()

                if not success:

                    time.sleep(1)

                    continue

            # ------------------------------------------------
            # Read frame
            # ------------------------------------------------

            success, frame = self.camera.read()

            if not success:

                self.status = "CAMERA ERROR"

                time.sleep(0.1)

                continue

            frame = cv2.flip(frame, 1)

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            current_time = time.time()

            delta = current_time - previous_time

            previous_time = current_time

            if delta > 0:

                instant_fps = 1.0 / delta

                self.fps = (
                    self.fps * 0.9
                    + instant_fps * 0.1
                )

            # ------------------------------------------------
            # MediaPipe
            # ------------------------------------------------

            data = self.detector.process(frame)

            self.face_detected = data["face_detected"]

            # ------------------------------------------------
            # NO FACE
            # ------------------------------------------------

            if not data["face_detected"]:

                self.eye_closed_frames = max(
                    0,
                    self.eye_closed_frames - 2
                )

                self.yawn_frames = max(
                    0,
                    self.yawn_frames - 2
                )

                self.nod_frames = max(
                    0,
                    self.nod_frames - 2
                )

                self.drowsiness_score = max(
                    0,
                    self.drowsiness_score - 2
                )

                self.status = "NO FACE DETECTED"

                self.alerts.stop_alarm()

                self.draw_no_face_warning(frame)

                self.publish_frame(frame)

                continue

            # ------------------------------------------------
            # Get measurements
            # ------------------------------------------------

            self.ear = data["ear"]
            self.mar = data["mar"]

            self.pitch = data["pitch"]
            self.yaw = data["yaw"]
            self.roll = data["roll"]

            # Draw MediaPipe landmarks

            self.detector.draw_landmarks(
                frame,
                data
            )

            # =================================================
            # EYE CLOSURE
            # =================================================

            if self.ear < self.ear_threshold:

                self.eye_closed_frames += 1

            else:

                self.eye_closed_frames = max(
                    0,
                    self.eye_closed_frames - 2
                )

            # =================================================
            # YAWN
            # =================================================

            if self.mar > MAR_THRESHOLD:

                self.yawn_frames += 1

            else:

                self.yawn_frames = max(
                    0,
                    self.yawn_frames - 2
                )

            # =================================================
            # HEAD NOD
            # =================================================

            if len(self.pitch_history) > 0:

                previous_pitch = self.pitch_history[-1]

                pitch_change = (
                    self.pitch - previous_pitch
                )

                if pitch_change > NOD_PITCH_CHANGE:

                    self.nod_frames += 1

                else:

                    self.nod_frames = max(
                        0,
                        self.nod_frames - 1
                    )

            self.pitch_history.append(
                self.pitch
            )

            # =================================================
            # CALCULATE DROWSINESS SCORE
            # =================================================

            eye_score = min(
                60,
                int(
                    60
                    * self.eye_closed_frames
                    / EAR_CONSECUTIVE_FRAMES
                )
            )

            yawn_score = min(
                20,
                int(
                    20
                    * self.yawn_frames
                    / YAWN_CONSECUTIVE_FRAMES
                )
            )

            nod_score = min(
                20,
                int(
                    20
                    * self.nod_frames
                    / NOD_CONSECUTIVE_FRAMES
                )
            )

            self.drowsiness_score = min(
                100,
                eye_score
                + yawn_score
                + nod_score
            )

            # =================================================
            # DROWSINESS DETECTION
            # =================================================

            eye_drowsiness = (
                self.eye_closed_frames
                >= EAR_CONSECUTIVE_FRAMES
            )

            yawn_nod_drowsiness = (
                self.yawn_frames
                >= YAWN_CONSECUTIVE_FRAMES
                and
                self.nod_frames
                >= NOD_CONSECUTIVE_FRAMES
            )

            drowsy = (
                eye_drowsiness
                or
                yawn_nod_drowsiness
                or
                self.drowsiness_score
                >= DROWSINESS_SCORE_THRESHOLD
            )

            # =================================================
            # ALERT
            # =================================================

            if drowsy:

                self.status = "DROWSINESS DETECTED"

                self.alerts.start_alarm()

                self.draw_drowsiness_warning(
                    frame
                )

            else:

                self.status = "NORMAL"

                self.alerts.stop_alarm()

            # =================================================
            # DRAW INFORMATION
            # =================================================

            self.draw_information(
                frame
            )

            # =================================================
            # PUBLISH FRAME
            # =================================================

            self.publish_frame(frame)


    # ========================================================
    # DRAW INFORMATION
    # ========================================================

    def draw_information(self, frame):

        cv2.rectangle(
            frame,
            (10, 10),
            (390, 205),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"EAR: {self.ear:.3f}",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"EAR Threshold: {self.ear_threshold:.3f}",
            (25, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"MAR: {self.mar:.3f}",
            (25, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Pitch: {self.pitch:.1f}",
            (25, 124),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Drowsiness Score: {self.drowsiness_score}/100",
            (25, 152),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (25, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


    # ========================================================
    # DROWSINESS WARNING
    # ========================================================

    def draw_drowsiness_warning(self, frame):

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (frame.shape[1], frame.shape[0]),
            (0, 0, 255),
            -1
        )

        frame[:] = cv2.addWeighted(
            overlay,
            0.18,
            frame,
            0.82,
            0
        )

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 70),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            "WARNING: DROWSINESS DETECTED",
            (30, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3
        )


    # ========================================================
    # NO FACE WARNING
    # ========================================================

    def draw_no_face_warning(self, frame):

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 65),
            (0, 120, 255),
            -1
        )

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (30, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )


    # ========================================================
    # PUBLISH JPEG
    # ========================================================

    def publish_frame(self, frame):

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                80
            ]
        )

        if not success:
            return

        with self.lock:

            self.latest_frame = buffer.tobytes()


    # ========================================================
    # GET STATUS
    # ========================================================

    def get_status(self):

        with self.lock:

            return {
                "status": self.status,
                "face_detected": self.face_detected,
                "ear": round(self.ear, 3),
                "mar": round(self.mar, 3),
                "pitch": round(self.pitch, 1),
                "yaw": round(self.yaw, 1),
                "roll": round(self.roll, 1),
                "ear_threshold": (
                    round(self.ear_threshold, 3)
                    if self.ear_threshold is not None
                    else None
                ),
                "score": self.drowsiness_score,
                "fps": round(self.fps, 1),
                "eye_frames": self.eye_closed_frames,
                "yawn_frames": self.yawn_frames,
                "nod_frames": self.nod_frames,
                "calibration_remaining": self.calibration_remaining
            }


# ============================================================
# CREATE SYSTEM
# ============================================================

system = DrowsinessWebSystem()


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template("index.html")


@app.route("/video_feed")
def video_feed():

    def generate():

        while True:

            with system.lock:

                frame = system.latest_frame

            if frame is not None:

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )

            time.sleep(0.03)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/status")
def api_status():

    return jsonify(
        system.get_status()
    )


@app.route("/api/recalibrate", methods=["POST"])
def api_recalibrate():

    system.request_recalibration()

    return jsonify({
        "success": True
    })


# ============================================================
# START SERVER
# ============================================================

def open_browser():

    webbrowser.open(
        "http://127.0.0.1:5000"
    )


if __name__ == "__main__":

    print("=" * 60)
    print("AI DRIVER DROWSINESS DETECTION")
    print("WEB VERSION")
    print("=" * 60)

    if not system.start():

        print("Could not start camera.")

    else:

        print("Camera started.")
        print("Starting web server...")

        threading.Timer(
            1.5,
            open_browser
        ).start()

    try:

        app.run(
            host="127.0.0.1",
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False
        )

    finally:

        system.stop()