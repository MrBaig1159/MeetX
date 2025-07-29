import cv2
import numpy as np
import pyautogui
import time
from input.kinect import KinectHandler
from processing.ir_hand import IRHandProcessor
from control.cursor import CursorController
from ui.overlay import OverlayRenderer
from utils.config import Config

class GestureMouseController:
    def __init__(self):
        self.config = Config()
        self.kinect = KinectHandler()
        self.ir_processor = IRHandProcessor(self.config)
        self.cursor = CursorController(self.config)
        self.overlay = OverlayRenderer(self.config)
        self.frame_count = 0
        self.frame_skip = 0
        self.fps_counter = time.time()
        self.fps_display = 0

    def run(self):
        if not self.kinect.initialize():
            print("Failed to initialize Kinect")
            return

        print("Starting Gesture Control. Press 'q' to quit.")
        cv2.namedWindow("Gesture Control", cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                rgb_frame, depth_frame = self.kinect.get_frames()
                if rgb_frame is None or depth_frame is None:
                    print("No frame from Kinect.")
                    break

                rgb_frame = cv2.flip(rgb_frame, 1)
                depth_frame = cv2.flip(depth_frame, 1)
                self.frame_count += 1
                self.frame_skip += 1

                current_time = time.time()
                if self.frame_count % 30 == 0:
                    elapsed = current_time - self.fps_counter
                    if elapsed > 0:
                        self.fps_display = 30 / elapsed
                    self.fps_counter = current_time

                if self.frame_skip < self.config.process_every_n_frames:
                    self.cursor.animate_to_target()
                    continue
                self.frame_skip = 0

                ir_result = self.ir_processor.process(rgb_frame, depth_frame)
                display_frame = self.overlay.render(rgb_frame.copy(), ir_result, self.fps_display, self.cursor)

                self.cursor.update(ir_result, rgb_frame.shape)

                cv2.imshow("Gesture Control", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                self._handle_key(key)
        finally:
            cv2.destroyAllWindows()

    def _handle_key(self, key):
        # Placeholder for hotkey handling (cycle modes, reset cursor, etc.)
        pass

if __name__ == "__main__":
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
    controller = GestureMouseController()
    controller.run()
