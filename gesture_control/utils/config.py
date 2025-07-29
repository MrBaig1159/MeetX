# utils/config.py

class Config:
    def __init__(self):
        # IR detection parameters
        self.IR_HAND_MIN_DEPTH = 400
        self.IR_HAND_MAX_DEPTH = 1200
        self.IR_BLOB_MIN_AREA = 500
        self.IR_BLOB_MAX_AREA = 5000
        self.IR_CONTOUR_APPROX_EPSILON = 0.02

        # Smoothing and cursor config
        self.SMOOTHING_FRAMES = 6
        self.SMOOTHING_FACTOR = 0.75
        self.DEADZONE_RADIUS = 3
        self.CLICK_THRESHOLD = 40
        self.CLICK_COOLDOWN = 0.3

        # Frame processing rate
        self.process_every_n_frames = 2
