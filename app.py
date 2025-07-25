import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import time
from collections import deque
import freenect
import threading
import subprocess
import os

# Enhanced IR-focused Configuration
CLICK_THRESHOLD = 40
SMOOTHING_FRAMES = 6
CLICK_COOLDOWN = 0.3
MOVEMENT_THRESHOLD = 4
SMOOTHING_FACTOR = 0.75
DEADZONE_RADIUS = 3

# IR Detection Parameters
IR_HAND_MIN_DEPTH = 400
IR_HAND_MAX_DEPTH = 1200
IR_BLOB_MIN_AREA = 500
IR_BLOB_MAX_AREA = 5000
IR_CONTOUR_APPROX_EPSILON = 0.02

# Enhanced Depth/IR Visualization Modes
DEPTH_MODES = {
    'RAINBOW': cv2.COLORMAP_RAINBOW,
    'JET': cv2.COLORMAP_JET,
    'HOT': cv2.COLORMAP_HOT,
    'COOL': cv2.COLORMAP_COOL,
    'VIRIDIS': cv2.COLORMAP_VIRIDIS,
    'PLASMA': cv2.COLORMAP_PLASMA,
    'MAGMA': cv2.COLORMAP_MAGMA,
    'INFERNO': cv2.COLORMAP_INFERNO,
    'PARULA': cv2.COLORMAP_PARULA,
    'BONE': cv2.COLORMAP_BONE,
    'GRAY': cv2.COLORMAP_GRAY
}

# Processing modes
PROCESSING_MODES = {
    'RGB_ONLY': 'RGB with MediaPipe only',
    'DEPTH_ASSISTED': 'Depth-assisted MediaPipe',
    'IR_PURE': 'Pure IR blob detection',
    'HYBRID': 'Hybrid IR + MediaPipe'
}

# Current modes
current_depth_mode = 'JET'
current_processing_mode = 'HYBRID'

# Depth filtering parameters
MIN_DEPTH = 400
MAX_DEPTH = 1200
DEPTH_THRESHOLD = 100

# Enhanced Mapping Configuration
MAPPING_MODE = "AMPLIFIED"
AMPLIFICATION_FACTOR = 1.3
CENTER_DAMPING = 0.8
DEADZONE_INNER = 0.2
DEADZONE_OUTER = 0.8
ELASTIC_STRENGTH = 0.4
CURVE_POWER = 1.5

# Initialize with optimized settings
screen_width, screen_height = pyautogui.size()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.4,  # Lower for IR-assisted detection
    min_tracking_confidence=0.4,   # Lower for IR-assisted detection
    model_complexity=0
)

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Tracking variables
position_buffer = deque(maxlen=SMOOTHING_FRAMES)
last_smooth_pos = np.array([screen_width//2, screen_height//2], dtype=np.float32)
current_cursor_pos = np.array([screen_width//2, screen_height//2], dtype=np.float32)
target_cursor_pos = np.array([screen_width//2, screen_height//2], dtype=np.float32)
last_click_time = 0
frame_count = 0
fps_counter = time.time()
is_animating = False

# GUI state variables
last_fps_display = 0
click_visual_timer = 0
click_visual_active = False
show_depth_overlay = False
show_ir_overlay = True
depth_overlay_alpha = 0.5
ir_overlay_alpha = 0.7
freenect_glview_process = None

# IR tracking state
ir_hand_detected = False
ir_hand_center = None
ir_fingertips = []
ir_confidence = 0.0

# Disable pyautogui failsafe
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

def get_kinect_data():
    """Get RGB and depth data from Kinect v1"""
    try:
        # Get RGB frame
        rgb_frame, _ = freenect.sync_get_video()
        
        # Get depth frame (this is derived from IR)
        depth_frame, _ = freenect.sync_get_depth()
        
        # Convert RGB from Kinect format to OpenCV format
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        return rgb_frame, depth_frame
    except Exception as e:
        print(f"Error getting Kinect data: {e}")
        return None, None

def create_ir_hand_mask(depth_frame, min_depth=IR_HAND_MIN_DEPTH, max_depth=IR_HAND_MAX_DEPTH):
    """Create a mask for hand detection using IR depth data"""
    # Create depth mask for hand region
    depth_mask = np.logical_and(depth_frame >= min_depth, depth_frame <= max_depth)
    
    # Convert to uint8 for processing
    mask = depth_mask.astype(np.uint8) * 255
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply Gaussian blur to smooth edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    return mask

def detect_ir_hand_blob(depth_frame, ir_mask):
    """Detect hand blob using IR depth data"""
    global ir_hand_detected, ir_hand_center, ir_fingertips, ir_confidence
    
    # Find contours in the IR mask
    contours, _ = cv2.findContours(ir_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        ir_hand_detected = False
        ir_hand_center = None
        ir_fingertips = []
        ir_confidence = 0.0
        return None, []
    
    # Find the largest contour (assuming it's the hand)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # Check if contour is within reasonable size range
    if area < IR_BLOB_MIN_AREA or area > IR_BLOB_MAX_AREA:
        ir_hand_detected = False
        ir_hand_center = None
        ir_fingertips = []
        ir_confidence = 0.0
        return None, []
    
    # Calculate confidence based on contour properties
    perimeter = cv2.arcLength(largest_contour, True)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        ir_confidence = min(1.0, area / IR_BLOB_MAX_AREA + circularity * 0.5)
    else:
        ir_confidence = 0.0
    
    # Get hand center
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        ir_hand_center = (cx, cy)
    else:
        ir_hand_center = None
    
    # Find fingertips using convex hull and defects
    fingertips = []
    
    # Approximate contour to reduce noise
    epsilon = IR_CONTOUR_APPROX_EPSILON * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Find convex hull
    hull = cv2.convexHull(approx, returnPoints=False)
    
    if len(hull) > 3:
        # Find convexity defects
        defects = cv2.convexityDefects(approx, hull)
        
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                
                # Calculate angles and distances to identify fingertips
                a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                
                # Use cosine rule to find angle
                if b != 0 and c != 0:
                    angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))
                    
                    # If angle is less than 90 degrees, it might be a fingertip
                    if angle <= np.pi/2:
                        fingertips.append(start)
    
    # Limit fingertips to reasonable number
    if len(fingertips) > 5:
        # Sort by distance from center and take closest 5
        if ir_hand_center:
            fingertips.sort(key=lambda p: np.sqrt((p[0] - ir_hand_center[0])**2 + (p[1] - ir_hand_center[1])**2))
            fingertips = fingertips[:5]
    
    ir_hand_detected = True
    ir_fingertips = fingertips
    
    return largest_contour, fingertips

def enhance_depth_visualization(depth_frame, mode='JET'):
    """Enhanced depth visualization with IR focus"""
    # Clip depth values to reasonable range
    depth_clipped = np.clip(depth_frame, MIN_DEPTH, MAX_DEPTH)
    
    # Normalize to 0-255 range
    depth_normalized = cv2.convertScaleAbs(depth_clipped, 
                                          alpha=255.0/(MAX_DEPTH-MIN_DEPTH), 
                                          beta=-255.0*MIN_DEPTH/(MAX_DEPTH-MIN_DEPTH))
    
    # Apply Gaussian blur for smoother visualization
    depth_smoothed = cv2.GaussianBlur(depth_normalized, (3, 3), 0)
    
    # Apply color map
    colormap = DEPTH_MODES.get(mode, cv2.COLORMAP_JET)
    depth_colored = cv2.applyColorMap(depth_smoothed, colormap)
    
    return depth_colored, depth_normalized

def process_hybrid_detection(rgb_frame, depth_frame):
    """Hybrid processing combining MediaPipe and IR detection"""
    # Create IR mask for hand region
    ir_mask = create_ir_hand_mask(depth_frame)
    
    # Detect hand blob using IR
    ir_contour, ir_fingertips = detect_ir_hand_blob(depth_frame, ir_mask)
    
    # Process with MediaPipe
    rgb_for_mp = rgb_frame.copy()
    
    # If we have IR detection, focus MediaPipe on that region
    if ir_hand_detected and ir_hand_center:
        # Create a mask to help MediaPipe focus
        mask_for_mp = np.zeros(rgb_frame.shape[:2], dtype=np.uint8)
        if ir_contour is not None:
            cv2.fillPoly(mask_for_mp, [ir_contour], 255)
            # Dilate to give some margin
            kernel = np.ones((20, 20), np.uint8)
            mask_for_mp = cv2.dilate(mask_for_mp, kernel, iterations=1)
        
        # Apply mask to RGB (darken non-hand areas)
        rgb_for_mp[mask_for_mp == 0] = rgb_for_mp[mask_for_mp == 0] * 0.3
    
    # Convert to RGB for MediaPipe
    rgb_mp = cv2.cvtColor(rgb_for_mp, cv2.COLOR_BGR2RGB)
    mp_result = hands.process(rgb_mp)
    
    return mp_result, ir_contour, ir_fingertips, ir_mask

def draw_ir_overlay(display_frame, ir_contour, ir_fingertips, ir_mask):
    """Draw IR detection overlay"""
    if not ir_hand_detected:
        return
    
    # Draw IR mask as overlay
    if show_ir_overlay:
        # Create colored mask
        ir_colored = np.zeros_like(display_frame)
        ir_colored[ir_mask > 0] = [0, 255, 0]  # Green for IR detection
        
        # Blend with display frame
        display_frame = cv2.addWeighted(display_frame, 1-ir_overlay_alpha, 
                                      ir_colored, ir_overlay_alpha, 0)
    
    # Draw hand contour
    if ir_contour is not None:
        cv2.drawContours(display_frame, [ir_contour], -1, (0, 255, 0), 2)
    
    # Draw hand center
    if ir_hand_center:
        cv2.circle(display_frame, ir_hand_center, 8, (0, 255, 0), -1)
        cv2.putText(display_frame, f"IR: {ir_confidence:.2f}", 
                   (ir_hand_center[0] + 10, ir_hand_center[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw IR fingertips
    for i, fingertip in enumerate(ir_fingertips):
        cv2.circle(display_frame, fingertip, 6, (255, 0, 0), -1)
        cv2.putText(display_frame, f"F{i+1}", 
                   (fingertip[0] + 8, fingertip[1] - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    return display_frame

def draw_enhanced_ui(display_frame, current_time, pixel_distance=0):
    """Enhanced UI with IR information"""
    global last_fps_display, click_visual_timer, click_visual_active
    
    # Update FPS display
    if frame_count % 30 == 0:
        fps = 30 / (current_time - fps_counter) if current_time - fps_counter > 0 else 0
        last_fps_display = fps
    
    # Main UI background
    ui_height = 400
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (5, 5), (450, ui_height), (0, 0, 0), -1)
    cv2.rectangle(overlay, (5, 5), (450, ui_height), (50, 50, 50), 2)
    cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
    
    # Enhanced UI text
    y_offset = 30
    cv2.putText(display_frame, "Enhanced IR Kinect Mouse Control", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_offset += 30
    cv2.putText(display_frame, f"FPS: {last_fps_display:.1f}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    y_offset += 25
    cv2.putText(display_frame, f"Processing: {current_processing_mode}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    y_offset += 25
    cv2.putText(display_frame, f"Depth Mode: {current_depth_mode}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
    y_offset += 25
    cv2.putText(display_frame, f"IR Range: {MIN_DEPTH}-{MAX_DEPTH}mm", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 25
    
    # IR detection status
    ir_status = "DETECTED" if ir_hand_detected else "NOT DETECTED"
    ir_color = (0, 255, 0) if ir_hand_detected else (0, 0, 255)
    cv2.putText(display_frame, f"IR Hand: {ir_status}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ir_color, 1)
    y_offset += 25
    
    if ir_hand_detected:
        cv2.putText(display_frame, f"IR Confidence: {ir_confidence:.2f}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(display_frame, f"IR Fingertips: {len(ir_fingertips)}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    
    # Overlay status
    overlay_status = "ON" if show_ir_overlay else "OFF"
    cv2.putText(display_frame, f"IR Overlay: {overlay_status} ({ir_overlay_alpha:.1f})", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y_offset += 25
    
    # Cursor position
    cv2.putText(display_frame, f"Cursor: ({int(current_cursor_pos[0])}, {int(current_cursor_pos[1])})", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y_offset += 25
    
    # Pinch distance
    if pixel_distance > 0:
        color = (0, 255, 0) if pixel_distance < CLICK_THRESHOLD else (255, 255, 255)
        cv2.putText(display_frame, f"Pinch: {pixel_distance:.0f}px", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 25
    
    # Click visual feedback
    if click_visual_active:
        if current_time - click_visual_timer < 0.5:
            cv2.putText(display_frame, "CLICK!", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            click_visual_active = False
    
    # Help text
    help_text = [
        "Controls: q=quit, p=processing mode, v=depth mode",
        "i=IR overlay, +=alpha up, -=alpha down, r=reset",
        "Works in complete darkness using IR sensors!"
    ]
    
    for i, text in enumerate(help_text):
        cv2.putText(display_frame, text, (10, 350 + i*15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

# Include all the previous helper functions (smoothing, mapping, etc.)
def lightweight_smoothing(new_pos):
    """Simple but effective smoothing"""
    global last_smooth_pos
    
    new_pos = np.array(new_pos, dtype=np.float32)
    position_buffer.append(new_pos)
    
    if len(position_buffer) > 1:
        weights = np.linspace(0.5, 1.0, len(position_buffer))
        weights /= weights.sum()
        
        smoothed = np.zeros(2, dtype=np.float32)
        for i, pos in enumerate(position_buffer):
            smoothed += weights[i] * pos
    else:
        smoothed = new_pos
    
    result = SMOOTHING_FACTOR * smoothed + (1 - SMOOTHING_FACTOR) * last_smooth_pos
    
    movement = np.linalg.norm(result - last_smooth_pos)
    if movement < DEADZONE_RADIUS:
        result = last_smooth_pos
    
    last_smooth_pos = result
    return result.astype(int)

def animate_cursor_to_target():
    """Smooth cursor animation"""
    global current_cursor_pos, target_cursor_pos, is_animating
    
    distance = np.linalg.norm(target_cursor_pos - current_cursor_pos)
    
    if distance > 2:
        direction = target_cursor_pos - current_cursor_pos
        step = direction * 0.3
        current_cursor_pos += step
        pyautogui.moveTo(int(current_cursor_pos[0]), int(current_cursor_pos[1]), duration=0)
        is_animating = True
        return True
    else:
        current_cursor_pos = target_cursor_pos.copy()
        pyautogui.moveTo(int(current_cursor_pos[0]), int(current_cursor_pos[1]), duration=0)
        is_animating = False
        return False

def update_cursor_target(new_target):
    """Update cursor target"""
    global target_cursor_pos
    target_cursor_pos = np.array(new_target, dtype=np.float32)

def smart_map_to_screen(x, y, frame_width, frame_height):
    """Smart coordinate mapping"""
    screen_x = int(screen_width * x / frame_width)
    screen_y = int(screen_height * y / frame_height)
    screen_x = max(0, min(screen_width - 1, screen_x))
    screen_y = max(0, min(screen_height - 1, screen_y))
    return screen_x, screen_y

def calculate_distance(pos1, pos2):
    """Calculate distance between two points"""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    return (dx*dx + dy*dy) ** 0.5

def should_click(thumb_pos, index_pos, current_time):
    """Click detection"""
    global last_click_time
    
    distance = calculate_distance(thumb_pos, index_pos)
    time_since_last_click = current_time - last_click_time
    
    return (distance < CLICK_THRESHOLD and 
            time_since_last_click > CLICK_COOLDOWN)

def main():
    global last_click_time, frame_count, fps_counter, current_cursor_pos, target_cursor_pos
    global current_processing_mode, current_depth_mode, show_ir_overlay, ir_overlay_alpha
    global click_visual_timer, click_visual_active, MIN_DEPTH, MAX_DEPTH
    
    print("Starting Enhanced IR Kinect v1 Hand Mouse Control...")
    print("This version works in complete darkness using IR sensors!")
    print("Enhanced Controls:")
    print("  q - Quit")
    print("  p - Cycle processing modes")
    print("  v - Cycle depth visualization modes")
    print("  i - Toggle IR overlay")
    print("  + - Increase IR overlay alpha")
    print("  - - Decrease IR overlay alpha")
    print("  r - Reset cursor")
    print("  [ - Decrease IR range")
    print("  ] - Increase IR range")
    
    # Check Kinect availability
    try:
        freenect.sync_get_video()
        print("Kinect v1 detected and initialized")
        print("IR sensors active - works in complete darkness!")
    except Exception as e:
        print(f"Error: Could not initialize Kinect v1: {e}")
        return
    
    # Initialize variables
    frame_skip = 0
    process_every_n_frames = 2
    current_cursor_pos = np.array(pyautogui.position(), dtype=np.float32)
    target_cursor_pos = current_cursor_pos.copy()
    
    cv2.namedWindow("Enhanced IR Kinect Hand Mouse Control", cv2.WINDOW_AUTOSIZE)
    
    try:
        while True:
            # Get Kinect data
            rgb_frame, depth_frame = get_kinect_data()
            
            if rgb_frame is None or depth_frame is None:
                print("Failed to get frame from Kinect")
                break
            
            frame_count += 1
            frame_skip += 1
            
            # Flip frames
            rgb_frame = cv2.flip(rgb_frame, 1)
            depth_frame = cv2.flip(depth_frame, 1)
            
            # Skip frames for performance
            if frame_skip < process_every_n_frames:
                animate_cursor_to_target()
                continue
            
            frame_skip = 0
            frame_height, frame_width, _ = rgb_frame.shape
            
            # Choose processing mode
            if current_processing_mode == 'HYBRID':
                mp_result, ir_contour, ir_fingertips, ir_mask = process_hybrid_detection(rgb_frame, depth_frame)
            else:
                # Fallback to standard processing
                rgb_mp = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                mp_result = hands.process(rgb_mp)
                ir_contour, ir_fingertips, ir_mask = None, [], np.zeros(rgb_frame.shape[:2], dtype=np.uint8)
            
            # Prepare display frame
            display_frame = rgb_frame.copy()
            
            # Draw IR overlay
            if ir_hand_detected:
                display_frame = draw_ir_overlay(display_frame, ir_contour, ir_fingertips, ir_mask)
            
            current_time = time.time()
            pixel_distance = 0
            
            # Process MediaPipe results
            if mp_result.multi_hand_landmarks:
                for hand_landmarks in mp_result.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    landmarks = hand_landmarks.landmark
                    
                    # Get finger positions
                    index_tip = landmarks[8]
                    thumb_tip = landmarks[4]
                    
                    index_x = int(index_tip.x * frame_width)
                    index_y = int(index_tip.y * frame_height)
                    thumb_x = int(thumb_tip.x * frame_width)
                    thumb_y = int(thumb_tip.y * frame_height)
                    
                    # Enhanced fingertip visualization
                    cv2.circle(display_frame, (index_x, index_y), 8, (0, 255, 255), -1)
                    cv2.circle(display_frame, (thumb_x, thumb_y), 8, (255, 0, 255), -1)
                    cv2.line(display_frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)
                    
                    # Cursor control
                    screen_x, screen_y = smart_map_to_screen(index_x, index_y, frame_width, frame_height)
                    smooth_x, smooth_y = lightweight_smoothing([screen_x, screen_y])
                    update_cursor_target([smooth_x, smooth_y])
                    animate_cursor_to_target()
                    
                    # Click detection
                    pixel_distance = calculate_distance((thumb_x, thumb_y), (index_x, index_y))
                    if should_click((thumb_x, thumb_y), (index_x, index_y), current_time):
                        pyautogui.click()
                        last_click_time = time.time()
                        click_visual_timer = time.time()
                        click_visual_active = True
                        cv2.circle(display_frame, (index_x, index_y), 20, (0, 0, 255), 3)
            
            # Fallback to IR-only control if MediaPipe fails
            elif ir_hand_detected and ir_hand_center and len(ir_fingertips) > 0:
                # Use IR center for cursor control
                screen_x, screen_y = smart_map_to_screen(ir_hand_center[0], ir_hand_center[1], 
                                                        frame_width, frame_height)
                smooth_x, smooth_y = lightweight_smoothing([screen_x, screen_y])
                update_cursor_target([smooth_x, smooth_y])
                animate_cursor_to_target()
                
                # Basic click detection based on IR fingertips
                if len(ir_fingertips) >= 2:
                    pixel_distance = calculate_distance(ir_fingertips[0], ir_fingertips[1])
                    if (pixel_distance < CLICK_THRESHOLD and 
                        current_time - last_click_time > CLICK_COOLDOWN):
                        pyautogui.click()
                        last_click_time = time.time()
                        click_visual_timer = time.time()
                        click_visual_active = True
            else:
                animate_cursor_to_target()
            
            # Draw enhanced UI
            draw_enhanced_ui(display_frame, current_time, pixel_distance)
            
            # Display frame
            cv2.imshow("Enhanced IR Kinect Hand Mouse Control", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                # Cycle processing modes
                modes = list(PROCESSING_MODES.keys())
                current_idx = modes.index(current_processing_mode)
                current_processing_mode = modes[(current_idx + 1) % len(modes)]
                print(f"Processing mode: {current_processing_mode} - {PROCESSING_MODES[current_processing_mode]}")
            elif key == ord('v'):
                # Cycle depth modes
                modes = list(DEPTH_MODES.keys())
                current_idx = modes.index(current_depth_mode)
                current_depth_mode = modes[(current_idx + 1) % len(modes)]
                print(f"Depth mode: {current_depth_mode}")
            elif key == ord('i'):
                show_ir_overlay = not show_ir_overlay
                print(f"IR overlay: {'ON' if show_ir_overlay else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                ir_overlay_alpha = min(1.0, ir_overlay_alpha + 0.1)
                print(f"IR overlay alpha: {ir_overlay_alpha:.1f}")
            elif key == ord('-') or key == ord('_'):
                ir_overlay_alpha = max(0.0, ir_overlay_alpha - 0.1)
                print(f"IR overlay alpha: {ir_overlay_alpha:.1f}")
            elif key == ord('['):
                MIN_DEPTH = max(200, MIN_DEPTH - 50)
                MAX_DEPTH = max(MIN_DEPTH + 100, MAX_DEPTH - 50)
                print(f"IR range: {MIN_DEPTH}-{MAX_DEPTH}mm (closer)")
            elif key == ord(']'):
                MIN_DEPTH = min(1000, MIN_DEPTH + 50)
                MAX_DEPTH = min(3000, MAX_DEPTH + 50)
                print(f"IR range: {MIN_DEPTH}-{MAX_DEPTH}mm (farther)")
            elif key == ord('r'):
                center_x, center_y = screen_width // 2, screen_height // 2
                current_cursor_pos = np.array([center_x, center_y], dtype=np.float32)
                target_cursor_pos = current_cursor_pos.copy()
                pyautogui.moveTo(center_x, center_y)
                print("Cursor reset to center")
            elif key == ord('h'):
                print("\n=== IR ENHANCED HELP ===")
                print("q - Quit")
                print("p - Cycle processing modes:")
                for mode, desc in PROCESSING_MODES.items():
                    print(f"    {mode}: {desc}")
                print("v - Cycle depth visualization modes")
                print("i - Toggle IR overlay")
                print("+ - Increase IR overlay transparency")
                print("- - Decrease IR overlay transparency")
                print("[ - Decrease IR detection range (closer)")
                print("] - Increase IR detection range (farther)")
                print("r - Reset cursor to center")
                print("h - Show this help")
                print("\nIR Features:")
                print("- Works in complete darkness")
                print("- Uses Kinect's IR projector and camera")
                print("- Depth-based hand detection")
                print("- Lighting-independent operation")
                print("========================")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("Cleanup complete")

if __name__ == "__main__":
    main()
    