# ui/overlay.py

import cv2
import numpy as np

class OverlayRenderer:
    def __init__(self, config):
        self.config = config
        self.show_ir_overlay = True
        self.ir_overlay_alpha = 0.7

    def render(self, frame, ir_result, fps_display, cursor):
        if self.show_ir_overlay and ir_result['ir_mask'] is not None:
            overlay = np.zeros_like(frame)
            overlay[ir_result['ir_mask'] > 0] = [0, 255, 0]
            frame = cv2.addWeighted(frame, 1 - self.ir_overlay_alpha, overlay, self.ir_overlay_alpha, 0)

        if ir_result['ir_contour'] is not None:
            cv2.drawContours(frame, [ir_result['ir_contour']], -1, (0, 255, 0), 2)

        if ir_result['ir_center']:
            cx, cy = ir_result['ir_center']
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(frame, f"IR: {ir_result['ir_confidence']:.2f}", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for i, tip in enumerate(ir_result['ir_fingertips']):
            cv2.circle(frame, tip, 6, (255, 0, 0), -1)
            cv2.putText(frame, f"F{i+1}", (tip[0] + 8, tip[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        self._draw_ui(frame, fps_display, ir_result, cursor)
        return frame

    def _draw_ui(self, frame, fps, ir_result, cursor):
        y = 30
        cv2.putText(frame, "IR Kinect Hand Mouse", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 25
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += 25
        cv2.putText(frame, f"IR Overlay: {'ON' if self.show_ir_overlay else 'OFF'} ({self.ir_overlay_alpha:.1f})",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y += 25

        if ir_result['ir_detected']:
            cv2.putText(frame, f"IR Hand: DETECTED", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 25
            cv2.putText(frame, f"Fingertips: {len(ir_result['ir_fingertips'])}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "IR Hand: NOT DETECTED", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y += 30

        pos = cursor.current_pos.astype(int)
        cv2.putText(frame, f"Cursor: ({pos[0]}, {pos[1]})", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)