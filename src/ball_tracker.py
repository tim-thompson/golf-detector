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
        self.last_known_static_pos = None  # Store the last position where ball was static
        
        # Motion detection parameters
        self.motion_threshold = 15        # Reduced threshold to detect subtle motion
        self.min_movement = 5
        self.motion_search_height = 0.8   # Search 80% of the frame height above the ball
        self.motion_memory_frames = 3     # Number of frames to remember motion
        self.recent_motion_areas = []     # Store recent areas of significant motion
    
    def _find_ball_in_search_area(self, frame: np.ndarray, search_area: np.ndarray, 
                                 x_offset: int, y_offset: int, debug_frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Dict]:
        """Find the most likely ball candidate in the search area and return debug info."""
        height = frame.shape[0]  # Get frame height for coordinate translation
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
        best_local_pos = None  # Store local position for debug views
        
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
            
            # Calculate local coordinates within search area
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
            consistency_score = 1.0 - min(1.0, brightness_std / 128.0)  # Clamp to [0,1]
            size_score = 1.0 - min(1.0, abs(area - np.pi * ((self.min_ball_size + self.max_ball_size)/4)**2) / 50.0)  # Normalize size difference
            
            # Heavily weight brightness and consistency for white ball detection
            score = (brightness_score * 0.6 +      # Increased weight for brightness
                    consistency_score * 0.25 +     # Slight decrease for consistency
                    circularity * 0.1 +           # Keep shape importance
                    size_score * 0.05)            # Reduced size importance
            
            # Draw valid candidates in green with detailed scores
            cv2.drawContours(debug_info['candidates'], [contour], -1, (0, 255, 0), 1)
            
            # Add more detailed scoring information with background for visibility
            score_info = [
                (f"B:{brightness_score:.2f} C:{consistency_score:.2f}", -15),
                (f"Circ:{circularity:.2f} S:{size_score:.2f}", 0),
                (f"Score:{score:.2f}", 15)
            ]
            
            for text, y_offset in score_info:
                # Add black background for text
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(debug_info['candidates'],
                            (cx - text_w//2 - 2, cy + y_offset - text_h - 2),
                            (cx + text_w//2 + 2, cy + y_offset + 2),
                            (0, 0, 0), -1)
                # Add centered text
                cv2.putText(debug_info['candidates'], text,
                          (cx - text_w//2, cy + y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Print debug info for high brightness candidates
            if brightness_score > 0.7:
                print(f"Candidate - Area: {area:.1f}, Brightness: {brightness_score:.2f}, " +
                      f"Consistency: {consistency_score:.2f}, Circularity: {circularity:.2f}, " +
                      f"Size Score: {size_score:.2f}, Final Score: {score:.2f}")
            
            # Update best candidate if score is higher
            if score > best_score:
                best_score = score
                best_local_pos = (cx, cy)  # Store local position
                # Translate to global coordinates - ensure we add the offsets correctly
                global_x = x_offset + cx
                global_y = frame.shape[0] - search_area.shape[0] - 10 + cy  # Calculate from bottom of frame
                best_candidate = (global_x, global_y)
                
                # Draw debug visualization in the candidates view
                cv2.drawContours(debug_info['candidates'], [contour], -1, (0, 255, 0), 2)
                
                # Draw the ball position in the main view
                cv2.circle(debug_frame, best_candidate, 5, (0, 255, 255), -1)  # Yellow dot
                cv2.circle(debug_frame, best_candidate, 8, (255, 0, 0), 2)     # Blue circle
                
                print(f"New best candidate! Score: {score:.2f}, Local pos: ({cx}, {cy}), " +
                      f"Global pos: ({global_x}, {global_y}), Search area offset: ({x_offset}, {y_offset})")
        
        # If we found a best candidate, highlight it in the debug views
        if best_local_pos:
            for view in ['grayscale', 'threshold', 'candidates']:
                # Draw a more visible circle
                cv2.circle(debug_info[view], best_local_pos, 10, (0, 0, 0), 3)  # Black outline
                cv2.circle(debug_info[view], best_local_pos, 10, (0, 255, 0), 2)  # Green circle
            print(f"Final best candidate - Local pos: {best_local_pos}, Global pos: {best_candidate}, Score: {best_score:.2f}")
        else:
            print("No valid ball candidate found")
        
        return best_candidate, debug_info
    
    def _detect_motion(self, current_frame: np.ndarray, prev_frame: np.ndarray, 
                      ball_pos: Tuple[int, int], search_radius: int) -> Tuple[bool, Dict]:
        """Detect if there is significant motion around the ball position."""
        if prev_frame is None:
            return False, {'motion_diff': None}
        
        height, width = current_frame.shape[:2]
        x, y = ball_pos
        
        # Define search area above the ball
        search_height = int(height * self.motion_search_height)
        y_start = max(0, y - search_height)  # Start from above the ball
        y_end = min(height, y + search_radius)  # End just below the ball
        x_start = max(0, x - search_radius * 2)  # Double the width of search area
        x_end = min(width, x + search_radius * 2)
        
        # Extract region for motion detection
        roi_current = current_frame[y_start:y_end, x_start:x_end]
        roi_prev = prev_frame[y_start:y_end, x_start:x_end]
        
        if roi_current.size == 0 or roi_prev.size == 0:
            return False, {'motion_diff': None}
        
        # Calculate absolute difference
        diff = cv2.absdiff(roi_current, roi_prev)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to motion difference
        _, motion_mask = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Find contours in motion mask
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create motion visualization - start with original frame
        motion_diff = roi_current.copy()
        
        # Draw search area with thick red border
        cv2.rectangle(motion_diff, (0, 0), (motion_diff.shape[1]-1, motion_diff.shape[0]-1), 
                     (0, 0, 255), 3)
        
        # Add "Motion Detection" label with black background
        label_bg = np.zeros((40, motion_diff.shape[1], 3), dtype=np.uint8)
        motion_diff[0:40, :] = label_bg
        cv2.putText(motion_diff, "Motion Detection", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Draw motion contours in blue with transparency
        motion_overlay = np.zeros_like(motion_diff)
        cv2.drawContours(motion_overlay, contours, -1, (255, 0, 0), -1)
        cv2.addWeighted(motion_diff, 1, motion_overlay, 0.3, 0, motion_diff)
        
        # Calculate mean motion and add to visualization
        mean_motion = np.mean(gray_diff)
        cv2.putText(motion_diff, f"Motion: {mean_motion:.1f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw recent motion trail with frame numbers
        for i in range(1, len(self.recent_motion_areas)):
            p1 = self.recent_motion_areas[i-1]
            p2 = self.recent_motion_areas[i]
            # Convert to local coordinates
            p1_local = (p1[0]-x_start, p1[1]-y_start)
            p2_local = (p2[0]-x_start, p2[1]-y_start)
            
            # Draw thick white line with black border
            cv2.line(motion_diff, p1_local, p2_local, (0, 0, 0), 4)  # Border
            cv2.line(motion_diff, p1_local, p2_local, (255, 255, 255), 2)  # Line
            
            # Draw numbered points
            cv2.circle(motion_diff, p2_local, 8, (0, 0, 0), -1)  # Black border
            cv2.circle(motion_diff, p2_local, 6, (255, 255, 0), -1)  # Yellow point
            cv2.putText(motion_diff, f"{i}", 
                       (p2_local[0]-4, p2_local[1]+4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
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
        
        # Static ball detection code
        search_width = int(width * self.search_width_percent)
        search_height = int(height * self.search_height_percent)
        right_offset = int(width * self.search_right_offset_percent)
        
        # Calculate search area position
        x_start = (width // 2 - search_width // 2) + right_offset
        x_start = min(x_start, width - search_width)  # Ensure we don't go past right edge
        x_start = max(x_start, 0)  # Ensure we don't go past left edge
        y_start = height - search_height - 10  # Position search area near bottom of frame
        
        # Extract search area
        search_area = frame[y_start:y_start + search_height, x_start:x_start + search_width]
        
        # Draw search area rectangle with label
        cv2.rectangle(debug_frame, (x_start, y_start), 
                     (x_start + search_width, y_start + search_height),
                     (0, 255, 0), 2)
        cv2.putText(debug_frame, "Ball Search Area", 
                   (x_start, y_start - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Ball detection - pass the correct offsets for coordinate translation
        ball_pos, detection_debug = self._find_ball_in_search_area(frame, search_area, x_start, y_start, debug_frame)
        
        # Print debug info about search area and ball position
        if ball_pos:
            print(f"Search area: x={x_start}, y={y_start}, w={search_width}, h={search_height}")
            print(f"Ball detected at: {ball_pos}")
        
        debug_info.update({
            'threshold_view': detection_debug['threshold'],
            'grayscale_view': detection_debug['grayscale'],
            'candidates_view': detection_debug['candidates']
        })
        
        # Always perform motion detection if we have a previous frame
        if self.prev_frame is not None:
            search_pos = ball_pos if ball_pos else self.last_known_static_pos
            if search_pos is None:
                # If no ball position, use center of search area
                search_pos = (x_start + search_width//2, y_start + search_height//2)
            
            # Check for motion
            is_moving, motion_debug = self._detect_motion(frame, self.prev_frame, search_pos, 
                                                        search_radius=50 if self.is_moving else 20)
            
            if motion_debug['motion_diff'] is not None:
                debug_info['motion_view'] = motion_debug['motion_diff']
            
            # Update motion state
            if ball_pos:
                if is_moving and not self.is_moving:
                    # Just started moving
                    self.is_moving = True
                    self.trajectory = [ball_pos]
                    # Draw "MOTION DETECTED" on main view
                    cv2.putText(debug_frame, "MOTION DETECTED!", 
                              (width//2 - 100, height//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
                if self.is_moving:
                    self.trajectory.append(ball_pos)
                    self.predicted_trajectory = self._predict_trajectory(self.trajectory)
                else:
                    self.last_known_static_pos = ball_pos
        
        # Update ball position
        self.ball_pos = ball_pos
        
        # Draw motion search area on main view
        if self.last_known_static_pos:
            x, y = self.last_known_static_pos
            search_height = int(height * self.motion_search_height)
            search_radius = 50 if self.is_moving else 20
            cv2.rectangle(debug_frame,
                         (x - search_radius, max(0, y - search_height)),
                         (x + search_radius, min(height, y + search_radius)),
                         (0, 0, 255), 2)
            cv2.putText(debug_frame, "Motion Search Area",
                       (x - search_radius, max(20, y - search_height - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw trajectory on main view if we have one
        if len(self.trajectory) > 1:
            # Draw trajectory line with thicker border for visibility
            for i in range(1, len(self.trajectory)):
                # Draw black border
                cv2.line(debug_frame, 
                        self.trajectory[i-1],
                        self.trajectory[i],
                        (0, 0, 0), 4)  # Thicker black border
                # Draw green line
                cv2.line(debug_frame, 
                        self.trajectory[i-1],
                        self.trajectory[i],
                        (0, 255, 0), 2)
            
            # Draw points and frame numbers along trajectory
            for i, point in enumerate(self.trajectory):
                # Draw point with black border
                cv2.circle(debug_frame, point, 5, (0, 0, 0), -1)  # Black border
                cv2.circle(debug_frame, point, 4, (0, 255, 255), -1)  # Yellow point
                
                # Draw frame number with black outline for visibility
                text = f"{i}"
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.putText(debug_frame, text, 
                          (point[0] - text_w//2, point[1] - 6),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)  # Black outline
                cv2.putText(debug_frame, text, 
                          (point[0] - text_w//2, point[1] - 6),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)  # Yellow text
            
            # Draw "TRACKING" status
            cv2.putText(debug_frame, "TRACKING BALL", 
                      (width//2 - 80, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
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
        self.last_known_static_pos = None
        self.recent_motion_areas = [] 