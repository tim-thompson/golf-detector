import cv2
import numpy as np
import torch
from typing import Tuple, List, Optional

class BallTracker:
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=15,
            varThreshold=25,
            detectShadows=False
        )
        
        # Parameters for ball detection
        self.min_ball_size = 5
        self.max_ball_size = 30
        self.ball_threshold = 240
        
        # Kalman filter for trajectory smoothing
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32) * 0.03
        
        self.prev_ball_pos = None
        self.trajectory = []
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for ball detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(gray)
        
        # Noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced, fg_mask
    
    def detect_ball(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect golf ball in the frame using multiple techniques."""
        enhanced, fg_mask = self.preprocess_frame(frame)
        
        # Threshold to isolate bright objects
        _, thresh = cv2.threshold(enhanced, self.ball_threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(thresh, fg_mask)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        min_diff = float('inf')
        
        for contour in contours:
            # Filter by area and circularity
            area = cv2.contourArea(contour)
            if area < np.pi * (self.min_ball_size/2)**2 or area > np.pi * (self.max_ball_size/2)**2:
                continue
                
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < 0.8:  # Not circular enough
                continue
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # If we have a previous position, prefer candidates closer to expected position
            if self.prev_ball_pos is not None:
                diff = np.sqrt((cx - self.prev_ball_pos[0])**2 + (cy - self.prev_ball_pos[1])**2)
                if diff < min_diff:
                    min_diff = diff
                    best_candidate = (cx, cy)
            else:
                best_candidate = (cx, cy)
                break
        
        if best_candidate is not None:
            # Update Kalman filter
            if self.prev_ball_pos is None:
                self.kalman.statePre = np.array([[best_candidate[0]], [best_candidate[1]], [0], [0]], np.float32)
            
            measurement = np.array([[best_candidate[0]], [best_candidate[1]]], np.float32)
            self.kalman.correct(measurement)
            prediction = self.kalman.predict()
            
            # Use smoothed position
            smoothed_pos = (int(prediction[0][0]), int(prediction[1][0]))
            self.prev_ball_pos = smoothed_pos
            self.trajectory.append(smoothed_pos)
            return smoothed_pos
        
        self.prev_ball_pos = None
        return None
    
    def get_trajectory(self) -> List[Tuple[int, int]]:
        """Return the current ball trajectory."""
        return self.trajectory
    
    def reset(self):
        """Reset the tracker state."""
        self.prev_ball_pos = None
        self.trajectory = []
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=15,
            varThreshold=25,
            detectShadows=False
        ) 