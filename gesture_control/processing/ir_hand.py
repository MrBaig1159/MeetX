# processing/ir_hand.py

import cv2
import numpy as np
import mediapipe as mp

class IRHandProcessor:
    def __init__(self, config):
        self.config = config
        self.ir_hand_detected = False
        self.ir_hand_center = None
        self.ir_fingertips = []
        self.ir_confidence = 0.0

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
            model_complexity=0
        )

    def process(self, rgb_frame, depth_frame):
        ir_mask = self._create_ir_mask(depth_frame)
        contour, fingertips = self._detect_ir_blob(depth_frame, ir_mask)

        if self.ir_hand_detected and self.ir_hand_center:
            rgb_for_mp = self._mask_rgb(rgb_frame, contour)
        else:
            rgb_for_mp = rgb_frame.copy()

        mp_result = self._run_mediapipe(rgb_for_mp)
        return {
            'mp_result': mp_result,
            'ir_mask': ir_mask,
            'ir_contour': contour,
            'ir_fingertips': fingertips,
            'ir_center': self.ir_hand_center,
            'ir_detected': self.ir_hand_detected,
            'ir_confidence': self.ir_confidence
        }

    def _create_ir_mask(self, depth):
        mask = np.logical_and(depth >= self.config.IR_HAND_MIN_DEPTH,
                              depth <= self.config.IR_HAND_MAX_DEPTH).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return mask

    def _detect_ir_blob(self, depth, mask):
        self.ir_hand_detected = False
        self.ir_hand_center = None
        self.ir_fingertips = []
        self.ir_confidence = 0.0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, []

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < self.config.IR_BLOB_MIN_AREA or area > self.config.IR_BLOB_MAX_AREA:
            return None, []

        peri = cv2.arcLength(largest, True)
        if peri > 0:
            circularity = 4 * np.pi * area / (peri * peri)
            self.ir_confidence = min(1.0, area / self.config.IR_BLOB_MAX_AREA + circularity * 0.5)

        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            self.ir_hand_center = (cx, cy)

        approx = cv2.approxPolyDP(largest, self.config.IR_CONTOUR_APPROX_EPSILON * peri, True)
        hull = cv2.convexHull(approx, returnPoints=False)
        fingertips = []

        if len(hull) > 3:
            defects = cv2.convexityDefects(approx, hull)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(approx[s][0])
                    end = tuple(approx[e][0])
                    far = tuple(approx[f][0])

                    a = np.linalg.norm(np.array(end) - np.array(start))
                    b = np.linalg.norm(np.array(far) - np.array(start))
                    c = np.linalg.norm(np.array(end) - np.array(far))

                    if b != 0 and c != 0:
                        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                        if angle <= np.pi / 2:
                            fingertips.append(start)

        if self.ir_hand_center and len(fingertips) > 5:
            fingertips.sort(key=lambda p: np.linalg.norm(np.array(p) - np.array(self.ir_hand_center)))
            fingertips = fingertips[:5]

        self.ir_hand_detected = True
        self.ir_fingertips = fingertips
        return largest, fingertips

    def _mask_rgb(self, rgb, contour):
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        kernel = np.ones((20, 20), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        rgb[mask == 0] = (rgb[mask == 0] * 0.3).astype(np.uint8)
        return rgb

    def _run_mediapipe(self, rgb):
        return self.hands.process(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
