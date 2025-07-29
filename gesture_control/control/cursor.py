# control/cursor.py

import pyautogui
import numpy as np
import time
from collections import deque

class CursorController:
    def __init__(self, config):
        self.config = config
        self.screen_width, self.screen_height = pyautogui.size()
        self.position_buffer = deque(maxlen=self.config.SMOOTHING_FRAMES)
        self.last_smooth_pos = np.array([self.screen_width // 2, self.screen_height // 2], dtype=np.float32)
        self.current_pos = self.last_smooth_pos.copy()
        self.target_pos = self.last_smooth_pos.copy()
        self.last_click_time = 0

    def update(self, ir_result, frame_shape):
        landmarks = ir_result['mp_result'].multi_hand_landmarks if ir_result['mp_result'] else None
        h, w = frame_shape[:2]
        pixel_distance = 0

        if landmarks:
            for hand_landmarks in landmarks:
                index = hand_landmarks.landmark[8]
                thumb = hand_landmarks.landmark[4]
                ix, iy = int(index.x * w), int(index.y * h)
                tx, ty = int(thumb.x * w), int(thumb.y * h)

                pixel_distance = self._distance((ix, iy), (tx, ty))
                smooth_x, smooth_y = self._smooth(*self._map(ix, iy, w, h))
                self._set_target([smooth_x, smooth_y])

                if self._should_click((ix, iy), (tx, ty)):
                    pyautogui.click()
                    self.last_click_time = time.time()
        elif ir_result['ir_detected'] and ir_result['ir_center']:
            cx, cy = ir_result['ir_center']
            smooth_x, smooth_y = self._smooth(*self._map(cx, cy, w, h))
            self._set_target([smooth_x, smooth_y])

            if len(ir_result['ir_fingertips']) >= 2:
                f1, f2 = ir_result['ir_fingertips'][:2]
                pixel_distance = self._distance(f1, f2)
                if pixel_distance < self.config.CLICK_THRESHOLD and self._click_ready():
                    pyautogui.click()
                    self.last_click_time = time.time()

        self.animate_to_target()

    def animate_to_target(self):
        dist = np.linalg.norm(self.target_pos - self.current_pos)
        if dist > 2:
            direction = self.target_pos - self.current_pos
            step = direction * 0.3
            self.current_pos += step
        else:
            self.current_pos = self.target_pos.copy()
        try:
            pyautogui.moveTo(int(self.current_pos[0]), int(self.current_pos[1]), duration=0)
        except:
            pass

    def _distance(self, p1, p2):
        dx, dy = p1[0] - p2[0], p1[1] - p2[1]
        return (dx**2 + dy**2) ** 0.5

    def _click_ready(self):
        return time.time() - self.last_click_time > self.config.CLICK_COOLDOWN

    def _map(self, x, y, fw, fh):
        sx = int(self.screen_width * x / fw)
        sy = int(self.screen_height * y / fh)
        return max(0, min(self.screen_width - 1, sx)), max(0, min(self.screen_height - 1, sy))

    def _smooth(self, x, y):
        point = np.array([x, y], dtype=np.float32)
        self.position_buffer.append(point)

        if len(self.position_buffer) > 1:
            weights = np.linspace(0.5, 1.0, len(self.position_buffer))
            weights /= weights.sum()
            smoothed = sum(w * p for w, p in zip(weights, self.position_buffer))
        else:
            smoothed = point

        result = self.config.SMOOTHING_FACTOR * smoothed + (1 - self.config.SMOOTHING_FACTOR) * self.last_smooth_pos
        if np.linalg.norm(result - self.last_smooth_pos) < self.config.DEADZONE_RADIUS:
            result = self.last_smooth_pos
        self.last_smooth_pos = result
        return result.astype(int)

    def _set_target(self, target):
        self.target_pos = np.array(target, dtype=np.float32)

    def reset_cursor(self):
        center = np.array([self.screen_width // 2, self.screen_height // 2], dtype=np.float32)
        self.current_pos = center.copy()
        self.target_pos = center.copy()
        pyautogui.moveTo(int(center[0]), int(center[1]))
