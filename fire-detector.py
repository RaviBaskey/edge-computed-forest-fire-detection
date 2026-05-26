"""
Edge-Computed Forest Fire Detection System
Phase 1: Heuristic-Based Vision Node Prototype

This script implements a real-time, zero-latency computer vision pipeline 
to detect forest fire plasma signatures. It uses a multi-layered heuristic
approach (Spectral Fusion + Morphological Gatekeeping) to ensure high 
sensitivity with minimal computational overhead.
"""

import cv2
import numpy as np
from pygame import mixer

# ==============================================================================
# CONFIGURATION & HARDWARE CONSTANTS
# ==============================================================================
INPUT_SOURCE = 0 
ALARM_AUDIO = 'alarm-sound.mp3'

# Detection thresholds tuned for Phase 1 prototype
F_THRESHOLD = 0.60          # Minimum red-dominance ratio
MIN_AREA = 100              # Minimum contour area to filter sensor noise
REQUIRED_FRAMES = 15        # Temporal buffer size to eliminate transient noise

# Morphological gatekeeping thresholds to differentiate fire from solid objects
COMPLEXITY_THRESHOLD = 12.0 # Detects chaotic/jagged fire boundaries (P^2 / Area)
SOLIDITY_MAX = 0.99         # Rejects solid, non-organic shapes

# System state management
alarm_active = False
consecutive_fire_frames = 0 

# ==============================================================================
# AUDIO NOTIFICATION HANDLER
# ==============================================================================
mixer.init()
try:
    mixer.music.load(ALARM_AUDIO) 
    mixer.music.set_volume(0.3)
except Exception as e:
    print(f"Audio Error: {e}. Ensure '{ALARM_AUDIO}' is present.")

# ==============================================================================
# CORE DETECTION ENGINE
# ==============================================================================
def process_frame(frame):
    """
    Processes a raw camera frame through a three-layer pipeline:
    1. Spectral Normalization (RGB/HSV Fusion)
    2. Morphological Gatekeeping (Geometry Analysis)
    3. Temporal Verification (State Machine)
    """
    global consecutive_fire_frames, alarm_active
    
    # Pre-processing: Downscale to reduce CPU load & Gaussian blur to suppress ISO noise
    frame_resized = cv2.resize(frame, (800, 450))
    blur = cv2.GaussianBlur(frame_resized, (15, 15), 0)

    # --- LAYER 1: SPECTRAL ANALYSIS ---
    bgr_float = blur.astype(np.float32)
    B, G, R = cv2.split(bgr_float)
    S = R + G + B + 1e-6 
    
    # Normalize RGB to achieve illumination-invariant detection
    R1, G1, B1 = R/S, G/S, B/S
    Y1 = (G1 + B1) / 2.0 
    red_dominant = (R1 >= G1) & (R1 >= B1)
    
    # Dark room failsafe: Ensure bright pixels are true fire, not white LEDs
    white_hot_core = (R > 220) & (G > 180) & (B < R - 40)
    rgb_mask = np.where(((np.maximum(R1, Y1) * red_dominant) > F_THRESHOLD) | white_hot_core, 255, 0).astype(np.uint8)

    # Layer 2: HSV fusion to track fire in deep shadows
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, np.array([0, 100, 150]), np.array([35, 255, 255]))

    # --- LAYER 3: FUSION & SPATIAL ANALYSIS ---
    binary_mask = cv2.dilate(cv2.bitwise_or(rgb_mask, hsv_mask), np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fire_detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA: 
            # Calculate geometric complexity and solidity for gatekeeping
            perimeter = cv2.arcLength(cnt, True)
            complexity = (perimeter * perimeter) / area if area > 0 else 0
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity = area / float(hull_area) if hull_area > 0 else 1.0
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h # Block wide artificial lights

            # Final validation: Organic fire = Jagged boundary + Gaps + Vertical Orientation
            if complexity >= COMPLEXITY_THRESHOLD and solidity <= SOLIDITY_MAX and aspect_ratio <= 1.5:
                cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 0, 255), 2)
                fire_detected = True
            else:
                cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # --- TEMPORAL STATE MACHINE ---
    if fire_detected:
        consecutive_fire_frames += 1
        if consecutive_fire_frames >= REQUIRED_FRAMES and not alarm_active:
            mixer.music.play(-1)
            alarm_active = True
    else:
        consecutive_fire_frames = 0 
        if alarm_active:
            mixer.music.stop()
            alarm_active = False

    cv2.imshow("Hardware Live Feed", frame_resized)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    video = cv2.VideoCapture(INPUT_SOURCE)
    # Hardware exposure lock: acts as "digital sunglasses" to preserve plasma color
    video.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
    video.set(cv2.CAP_PROP_EXPOSURE, -5) 

    print("Vision Node Active. Press 'q' to terminate.")
    while True:
        success, frame = video.read()
        if not success: break
        process_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()