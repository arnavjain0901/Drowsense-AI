# alerts.py

import csv
import os
import time
import math
import struct

import pygame


class AlertSystem:
    """
    Handles:

    - Audio alarm
    - Red visual warning
    - CSV event logging
    """

    def __init__(
        self,
        log_file="drowsiness_log.csv"
    ):

        self.log_file = log_file

        # -------------------------------------------------
        # Initialize pygame audio
        # -------------------------------------------------

        pygame.mixer.init(
            frequency=44100,
            size=-16,
            channels=1
        )

        # Generate alarm sound.
        self.alarm_sound = self._create_alarm()

        # Keep track of whether alarm is currently playing.
        self.alarm_playing = False

        # Create CSV log file.
        self._initialize_log()

    # -----------------------------------------------------
    # Create alarm sound
    # -----------------------------------------------------

    def _create_alarm(self):
        """
        Generate an alternating alarm tone.

        This means you don't need to download
        an external .wav or .mp3 file.
        """

        sample_rate = 44100

        duration = 0.5

        samples = int(
            sample_rate * duration
        )

        buffer = bytearray()

        for i in range(samples):

            t = i / sample_rate

            # Alternate between two frequencies.
            if int(t * 4) % 2 == 0:

                frequency = 1000

            else:

                frequency = 1400

            value = int(
                16000 *
                math.sin(
                    2 *
                    math.pi *
                    frequency *
                    t
                )
            )

            buffer.extend(
                struct.pack(
                    "<h",
                    value
                )
            )

        return pygame.mixer.Sound(
            buffer=bytes(buffer)
        )

    # -----------------------------------------------------
    # Initialize CSV file
    # -----------------------------------------------------

    def _initialize_log(self):

        # Only create the header if the file
        # doesn't already exist.

        if not os.path.exists(
            self.log_file
        ):

            with open(
                self.log_file,
                "w",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "timestamp",
                    "alert_type"
                ])

    # -----------------------------------------------------
    # Log event
    # -----------------------------------------------------

    def log_event(
        self,
        alert_type
    ):
        """
        Add an event to the CSV file.
        """

        timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with open(
            self.log_file,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                timestamp,
                alert_type
            ])

    # -----------------------------------------------------
    # Start alarm
    # -----------------------------------------------------

    def start_alarm(self):

        # Don't restart the alarm every frame.
        if not self.alarm_playing:

            self.alarm_sound.play(
                loops=-1
            )

            self.alarm_playing = True

            # Log only when the alarm starts.
            self.log_event(
                "DROWSINESS_ALARM"
            )

    # -----------------------------------------------------
    # Stop alarm
    # -----------------------------------------------------

    def stop_alarm(self):

        if self.alarm_playing:

            self.alarm_sound.stop()

            self.alarm_playing = False

    # -----------------------------------------------------
    # Draw warning overlay
    # -----------------------------------------------------

    @staticmethod
    def draw_warning(
        frame,
        message="DROWSINESS ALERT!"
    ):
        """
        Add a transparent red warning over the camera feed.
        """

        import cv2

        height, width = frame.shape[:2]

        # Create overlay.
        overlay = frame.copy()

        # Full-screen red layer.
        cv2.rectangle(
            overlay,
            (0, 0),
            (width, height),
            (0, 0, 255),
            -1
        )

        # Blend with original frame.
        frame[:] = cv2.addWeighted(
            overlay,
            0.20,
            frame,
            0.80,
            0
        )

        # Strong red banner at the top.
        cv2.rectangle(
            frame,
            (0, 0),
            (width, 100),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            message,
            (30, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )

        return frame

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    def close(self):

        self.stop_alarm()

        pygame.mixer.quit()