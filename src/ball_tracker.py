import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict

class BallTracker:
    def __init__(self):
        # Parameters for ball detection
        self.min_ball_size = 2     # Minimum diameter in pixels
        self.max_ball_size = 12    # Maximum diameter in pixels
        self.ball_threshold = 240
        self.search_width_percent = 0.3
        self.search_height_percent = 0.25
        
        # Search area position adjustments for right-handed golfers
        self.search_right_offset_percent = 0.1
        
        # Tracking state
        self.ball_pos = None
        self.prev_frame = None
        self.trajectory = []
        self.is_moving = False
        self.predicted_trajectory = []
        
        # Motion detection parameters
        self.motion_threshold = 30
        self.min_movement = 5
    
    def _find_ball_in_search_area(self, frame: np.ndarray, search_area: np.ndarray, 
                                 x_offset: int, y_offset: int) -> Tuple[Optional[Tuple[int, int]], Dict]:
        """Find the most likely ball candidate in the search area and return debug info."""
        # Convert to grayscale
        gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use simple thresholding to find bright objects
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        # Create debug visualizations
        debug_info = {
            'grayscale': cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR),
            'threshold': cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
            'candidates': search_area.copy()
        }
        
        # Add brightness histogram to grayscale view
        hist = cv2.calcHist([blurred], [0], None, [256], [0, 256])
        hist_h = 50
        hist_w = debug_info['grayscale'].shape[1]
        hist_img = np.zeros((hist_h, hist_w, 3), np.uint8)
        cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
        for i in range(256):
            cv2.line(hist_img, (int(i * hist_w/256), hist_h),
                    (int(i * hist_w/256), hist_h - int(hist[i])),
                    (0, 255, 0), 1)
        debug_info['grayscale'] = np.vstack([debug_info['grayscale'], hist_img])
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        best_score = 0
        
        # Draw all contours first in blue (all detected bright objects)
        cv2.drawContours(debug_info['candidates'], contours, -1, (255, 0, 0), 1)
        
        # Calculate area thresholds - using simple area rather than circle area
        min_area = 4  # Minimum area in pixels
        max_area = 100  # Maximum area in pixels
        cv2.putText(debug_info['candidates'], 
                   f"Size range: {min_area:.0f}-{max_area:.0f}px", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Calculate centroid for all contours
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Add area to all contours
            cv2.putText(debug_info['candidates'], f"A:{area:.0f}", 
                       (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Filter by area
            if area < min_area or area > max_area:
                # Draw rejected contours in red with reason
                cv2.drawContours(debug_info['candidates'], [contour], -1, (0, 0, 255), 1)
                if area < min_area:
                    cv2.putText(debug_info['candidates'], "too small", 
                              (cx, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                else:
                    cv2.putText(debug_info['candidates'], "too big", 
                              (cx, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < 0.7:
                # Draw non-circular contours in yellow with circularity score
                cv2.drawContours(debug_info['candidates'], [contour], -1, (0, 255, 255), 1)
                cv2.putText(debug_info['candidates'], f"circ:{circularity:.2f}", 
                          (cx, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                continue
            
            # Create a slightly larger mask for brightness checking
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, 2)
            
            # Check brightness statistics in the region
            mean_brightness = cv2.mean(blurred, mask=mask)[0]
            roi = cv2.bitwise_and(blurred, blurred, mask=mask)
            brightness_std = np.std(roi[mask > 0])
            
            # Adjust scoring to emphasize brightness more
            brightness_score = mean_brightness / 255.0
            consistency_score = 1.0 - (brightness_std / 128.0)
            size_score = 1.0 - abs(area - np.pi * ((self.min_ball_size + self.max_ball_size)/4)**2)
            
            # Heavily weight brightness and consistency for white ball detection
            score = (brightness_score * 0.6 +      # Increased weight for brightness
                    consistency_score * 0.25 +     # Slight decrease for consistency
                    circularity * 0.1 +           # Keep shape importance
                    size_score * 0.05)            # Reduced size importance
            
            # Draw valid candidates in green with detailed scores
            cv2.drawContours(debug_info['candidates'], [contour], -1, (0, 255, 0), 1)
            
            # Add more detailed scoring information
            cv2.putText(debug_info['candidates'], 
                       f"B:{brightness_score:.2f} C:{consistency_score:.2f}", 
                       (cx, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(debug_info['candidates'],
                       f"Circ:{circularity:.2f} S:{size_score:.2f}", 
                       (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(debug_info['candidates'],
                       f"Score:{score:.2f}", 
                       (cx, cy+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Highlight best candidate with thicker outline
            if score > best_score:
                best_score = score
                best_candidate = (cx + x_offset, cy + y_offset)
                # Draw thicker contour for best candidate
                cv2.drawContours(debug_info['candidates'], [contour], -1, (0, 255, 0), 2)
        
        # If we found a best candidate, highlight it in the debug views
        if best_candidate:
            rel_x = best_candidate[0] - x_offset
            rel_y = best_candidate[1] - y_offset
            for view in ['grayscale', 'threshold', 'candidates']:
                cv2.circle(debug_info[view], (rel_x, rel_y), 10, (0, 255, 0), 2)
        
        return best_candidate, debug_info
    
    def _detect_motion(self, current_frame: np.ndarray, prev_frame: np.ndarray, 
                      ball_pos: Tuple[int, int], search_radius: int) -> Tuple[bool, Dict]:
        """Detect if there is significant motion around the ball position."""
        if prev_frame is None:
            return False, {'motion_diff': None}
            
        # Extract region around ball position
        x, y = ball_pos
        roi_current = current_frame[max(0, y-search_radius):min(current_frame.shape[0], y+search_radius),
                                  max(0, x-search_radius):min(current_frame.shape[1], x+search_radius)]
        roi_prev = prev_frame[max(0, y-search_radius):min(prev_frame.shape[0], y+search_radius),
                            max(0, x-search_radius):min(prev_frame.shape[1], x+search_radius)]
        
        if roi_current.size == 0 or roi_prev.size == 0:
            return False, {'motion_diff': None}
            
        # Calculate absolute difference
        diff = cv2.absdiff(roi_current, roi_prev)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Create color-mapped difference for visualization
        motion_diff = cv2.applyColorMap(gray_diff, cv2.COLORMAP_JET)
        
        # Add mean motion value to visualization
        mean_motion = np.mean(gray_diff)
        cv2.putText(motion_diff, f"Motion: {mean_motion:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return mean_motion > self.motion_threshold, {'motion_diff': motion_diff}
    
    def _predict_trajectory(self, points: List[Tuple[int, int]], num_points: int = 10) -> List[Tuple[int, int]]:
        """Predict future trajectory points based on current trajectory."""
        if len(points) < 3:
            return []
            
        # Fit a quadratic curve to the points
        x_coords = np.array([p[0] for p in points])
        y_coords = np.array([p[1] for p in points])
        t = np.arange(len(points))
        
        # Fit x coordinates (linear)
        x_coef = np.polyfit(t, x_coords, 1)
        # Fit y coordinates (quadratic for parabolic motion)
        y_coef = np.polyfit(t, y_coords, 2)
        
        # Predict future points
        future_t = np.arange(len(points), len(points) + num_points)
        future_x = np.polyval(x_coef, future_t)
        future_y = np.polyval(y_coef, future_t)
        
        return [(int(x), int(y)) for x, y in zip(future_x, future_y)]
    
    def track_frame(self, frame: np.ndarray) -> Dict:
        """Track the ball in the current frame and return tracking information."""
        debug_frame = frame.copy()
        height, width = frame.shape[:2]
        
        debug_info = {
            'main_view': debug_frame,
            'threshold_view': None,
            'grayscale_view': None,
            'candidates_view': None,
            'motion_view': None
        }
        
        if not self.is_moving:
            # Define bottom-middle search area with right-hand bias
            search_width = int(width * self.search_width_percent)
            search_height = int(height * self.search_height_percent)
            
            # Calculate x_start with right offset
            right_offset = int(width * self.search_right_offset_percent)
            x_start = (width // 2 - search_width // 2) + right_offset
            
            # Ensure x_start doesn't go beyond frame bounds
            x_start = min(x_start, width - search_width)
            x_start = max(x_start, 0)
            
            # Set y_start to be at the bottom of the frame minus the search height
            y_start = height - search_height
            
            search_area = frame[y_start:height,  # Use full height to bottom
                              x_start:x_start + search_width]
            
            # Draw search area on debug frame
            cv2.rectangle(debug_frame, (x_start, y_start), 
                         (x_start + search_width, height),  # Extend to bottom
                         (0, 255, 0), 2)
            
            # Find ball in search area
            ball_pos, detection_debug = self._find_ball_in_search_area(frame, search_area, x_start, y_start)
            
            # Update debug info
            debug_info.update({
                'threshold_view': detection_debug['threshold'],
                'grayscale_view': detection_debug['grayscale'],
                'candidates_view': detection_debug['candidates']
            })
            
            if ball_pos:
                self.ball_pos = ball_pos
                # Check for motion if we have a previous frame
                if self.prev_frame is not None:
                    self.is_moving, motion_debug = self._detect_motion(frame, self.prev_frame, ball_pos, 20)
                    if motion_debug['motion_diff'] is not None:
                        debug_info['motion_view'] = motion_debug['motion_diff']
        else:
            # Ball is moving, track in a larger area around previous position
            if self.ball_pos:
                search_radius = 50
                x, y = self.ball_pos
                search_area = frame[max(0, y-search_radius):min(height, y+search_radius),
                                  max(0, x-search_radius):min(width, x+search_radius)]
                
                # Draw search area on debug frame
                cv2.rectangle(debug_frame, 
                            (max(0, x-search_radius), max(0, y-search_radius)),
                            (min(width, x+search_radius), min(height, y+search_radius)),
                            (0, 255, 0), 2)
                
                ball_pos, detection_debug = self._find_ball_in_search_area(
                    frame, search_area,
                    max(0, x-search_radius),
                    max(0, y-search_radius)
                )
                
                # Update debug info
                debug_info.update({
                    'threshold_view': detection_debug['threshold'],
                    'grayscale_view': detection_debug['grayscale'],
                    'candidates_view': detection_debug['candidates']
                })
                
                if ball_pos:
                    self.ball_pos = ball_pos
                    self.trajectory.append(ball_pos)
                    # Update predicted trajectory
                    self.predicted_trajectory = self._predict_trajectory(self.trajectory)
        
        # Store current frame for next iteration
        self.prev_frame = frame.copy()
        
        return {
            'ball_pos': self.ball_pos,
            'is_moving': self.is_moving,
            'trajectory': self.trajectory.copy(),
            'predicted_trajectory': self.predicted_trajectory.copy(),
            'debug_info': debug_info
        }
    
    def reset(self):
        """Reset the tracker state."""
        self.ball_pos = None
        self.prev_frame = None
        self.trajectory = []
        self.is_moving = False
        self.predicted_trajectory = [] 