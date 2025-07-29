# input/kinect.py

import freenect
import cv2
import numpy as np

class KinectHandler:
    def __init__(self):
        self.initialized = False

    def initialize(self):
        try:
            freenect.sync_get_video()
            self.initialized = True
            print("Kinect v1 detected and initialized")
            return True
        except Exception as e:
            print(f"Error initializing Kinect: {e}")
            return False

    def get_frames(self):
        try:
            rgb_frame, _ = freenect.sync_get_video()
            depth_frame, _ = freenect.sync_get_depth()
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            return rgb_frame, depth_frame
        except Exception as e:
            print(f"Error getting Kinect data: {e}")
            return None, None
