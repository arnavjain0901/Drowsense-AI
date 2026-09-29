# calibration.py

import time
import numpy as np
import cv2


class EARCalibrator:
    """
    Performs a short calibration period to determine
    the user's normal Eye Aspect Ratio (EAR).

    The user should look naturally at the camera
    during calibration.
    """

    def __init__(
        self,
        duration=5.0,
        threshold_factor=0.70,
        minimum_threshold=0.15,
        maximum_threshold=0.32
    ):

        self.duration = duration

        # Example:
        # normal EAR = 0.30
        # factor = 0.70
        # threshold = 0.21
        self.threshold_factor = threshold_factor

        # Safety limits so calibration cannot produce
        # an unrealistic threshold.
        self.minimum_threshold = minimum_threshold
        self.maximum_threshold = maximum_threshold

    def calibrate(
        self,
        detector,
        camera,
        frame_width,
        frame_height,
        display_callback
    ):
        """
        Run the calibration process.

        Returns:
            Personalized EAR threshold
            or None if calibration fails.
        """

        start_time = time.time()

        # Store valid EAR measurements.
        ear_values = []

        while True:

            elapsed = time.time() - start_time

            # Calibration finished.
            if elapsed >= self.duration:
                break

            ret, frame = camera.read()

            if not ret:
                continue

            # Process the frame using our detector.
            data = detector.process(frame)

            if data["face_detected"]:

                ear = data["ear"]

                # Ignore obviously invalid EAR values.
                #
                # Normal EAR is usually somewhere around
                # this range, although it varies between people.
                if 0.05 < ear < 0.60:

                    ear_values.append(ear)

            # Calculate remaining calibration time.
            remaining = max(
                0,
                self.duration - elapsed
            )

            # Update calibration display.
            frame = display_callback(
                frame,
                remaining,
                len(ear_values)
            )

            cv2.imshow(
                "Driver Drowsiness Detection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # Allow user to cancel.
            if key == ord("q"):

                return None

        # We need enough measurements to calculate
        # a reliable baseline.
        if len(ear_values) < 10:

            return None

        # Median is used instead of mean because it is
        # less affected by unusual frames/blinks.
        normal_ear = float(
            np.median(ear_values)
        )

        # Calculate personalized threshold.
        threshold = (
            normal_ear *
            self.threshold_factor
        )

        # Keep threshold within reasonable limits.
        threshold = max(
            self.minimum_threshold,
            min(
                self.maximum_threshold,
                threshold
            )
        )

        print(
            f"Normal EAR: {normal_ear:.3f}"
        )

        print(
            f"Personalized EAR threshold: "
            f"{threshold:.3f}"
        )

        return threshold