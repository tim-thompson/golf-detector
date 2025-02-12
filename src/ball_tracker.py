import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from trajectory_tracer import TrajectoryTracer

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
        self.predicted_points = []
        self.prev_predicted_points = []
        self.is_moving = False
        self.predicted_trajectory = []
        self.last_known_static_pos = None  # Store the last position where ball was static
        self.frames_since_static = 0       # Counter for frames since last static detection
        self.max_static_loss_frames = 2    # Reduced from 3 to 2 frames
        self.consecutive_misses = 0        # Counter for consecutive frames where ball was not detected
        self.max_consecutive_misses = 3    # Maximum consecutive misses before resetting
        self.frames_since_lost = 0         # Counter for frames since ball was last detected
        self.prediction_delay = 2          # Number of frames to wait before showing predictions
        
        # Motion detection parameters
        self.motion_threshold = 15        # Reduced threshold to detect subtle motion
        self.min_movement = 5
        self.motion_search_height = 0.8   # Search 80% of the frame height above the ball
        self.motion_memory_frames = 3     # Number of frames to remember motion
        self.recent_motion_areas = []     # Store recent areas of significant motion
        
        # Completed trajectory visualization
        self.completed_trajectory = []      # Store completed shot trajectory
        self.completed_predictions = []     # Store the final predictions when shot completes
        self.frames_since_completion = 0    # Counter for frames since shot completion
        self.show_completed_frames = 120    # Show completed trajectory for ~4 seconds at 30fps
        self.fade_start_frame = 90         # Start fading after 3 seconds
        
        # Add prediction visualization parameters
        self.prediction_display_frames = 30    # Show predictions for ~1 second at 30fps
        self.prediction_fade_start = 15        # Start fading predictions after 0.5 seconds
        self.frames_since_prediction = 0       # Counter for prediction display timing
        
        # Add trajectory tracer for smooth visualization
        self.trajectory_tracer = TrajectoryTracer()
        
        # Add output frame
        self.output_frame = None
    
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
    
    def _predict_next_points(self, points: List[Tuple[int, int]], num_points: int = 5) -> List[Tuple[int, int]]:
        """Predict next trajectory points based on current trajectory."""
        if len(points) < 2:
            return []
            
        # Use more points to calculate velocity and acceleration trends
        num_points_to_use = min(5, len(points))  # Use up to last 5 points
        points_for_calc = points[-num_points_to_use:]
        
        # If we have a predicted search area center, add it to the calculation points
        if len(self.trajectory) >= 2:
            # Calculate predicted search area center using last two points
            p1 = self.trajectory[-2]
            p2 = self.trajectory[-1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            gravity_factor = 2
            predicted_x = int(p2[0] + dx)
            predicted_y = int(p2[1] + dy + gravity_factor)
            points_for_calc.append((predicted_x, predicted_y))
        
        # Calculate velocities between consecutive points
        velocities = []
        for i in range(1, len(points_for_calc)):
            p1 = points_for_calc[i-1]
            p2 = points_for_calc[i]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            velocities.append((dx, dy))
        
        # Calculate average velocity and its trend
        avg_dx = sum(v[0] for v in velocities) / len(velocities)
        avg_dy = sum(v[1] for v in velocities) / len(velocities)
        
        # Calculate acceleration (change in velocity)
        dy_changes = []
        for i in range(1, len(velocities)):
            dy_changes.append(velocities[i][1] - velocities[i-1][1])
        
        # If we have enough points, calculate vertical acceleration trend
        if dy_changes:
            avg_dy_change = sum(dy_changes) / len(dy_changes)
        else:
            avg_dy_change = 2  # Default gravity factor if not enough points
        
        # Get the last point as starting position
        last_point = points_for_calc[-1]  # Use the last point including predicted center
        x, y = last_point
        
        # Initialize current velocity components
        current_dx = avg_dx
        current_dy = avg_dy
        
        predicted_points = []
        for i in range(num_points):
            # Update position
            x = int(x + current_dx)
            y = int(y + current_dy)
            predicted_points.append((x, y))
            
            # Update vertical velocity with acceleration trend
            current_dy += avg_dy_change
            
            # Slightly reduce horizontal velocity (air resistance simulation)
            current_dx *= 0.98
        
        return predicted_points
    
    def track_frame(self, frame: np.ndarray) -> Dict:
        """Track the ball in the current frame and return tracking information."""
        # Create both debug and clean output frames
        debug_frame = frame.copy()
        self.output_frame = frame.copy()
        
        debug_info = {
            'main_view': debug_frame,
            'threshold_view': None,
            'grayscale_view': None,
            'candidates_view': None,
            'motion_view': None
        }
        
        # Always check for static ball in the original search area
        search_width = int(frame.shape[1] * self.search_width_percent)
        search_height = int(frame.shape[0] * self.search_height_percent)
        right_offset = int(frame.shape[1] * self.search_right_offset_percent)
        
        x_start = (frame.shape[1] // 2 - search_width // 2) + right_offset
        x_start = min(x_start, frame.shape[1] - search_width)
        x_start = max(x_start, 0)
        y_start = frame.shape[0] - search_height - 10
        
        search_area = frame[y_start:y_start + search_height, x_start:x_start + search_width]
        
        # Draw search area rectangle
        cv2.rectangle(debug_frame, (x_start, y_start), 
                     (x_start + search_width, y_start + search_height),
                     (0, 255, 0), 2)
        cv2.putText(debug_frame, "Ball Search Area", 
                   (x_start, y_start - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        ball_pos, detection_debug = self._find_ball_in_search_area(frame, search_area, x_start, y_start, debug_frame)
        debug_info.update({
            'threshold_view': detection_debug['threshold'],
            'grayscale_view': detection_debug['grayscale'],
            'candidates_view': detection_debug['candidates']
        })
        
        # If we detect a ball in the static area, reset motion tracking
        if ball_pos:
            if self.is_moving:
                print("Ball detected in static area - resetting motion tracking")
                self.is_moving = False
                self.trajectory = []
                self.predicted_points = []
            self.frames_since_static = 0
            self.last_known_static_pos = ball_pos
        else:
            self.frames_since_static += 1
            
            # Start looking for motion immediately when we lose the static ball
            if self.last_known_static_pos and self.prev_frame is not None:
                motion_pos = self._track_moving_ball(frame, self.prev_frame, debug_frame, debug_info)
                
                # If we've lost the ball long enough and found motion, switch to motion tracking
                if self.frames_since_static > self.max_static_loss_frames and motion_pos:
                    print(f"Lost static ball for {self.frames_since_static} frames and found motion - switching to motion tracking")
                    self.is_moving = True
                    if not self.trajectory:  # Only initialize trajectory if it's empty
                        self.trajectory = [self.last_known_static_pos]
                    ball_pos = motion_pos
        
        # If we're in motion tracking mode, continue tracking
        if self.is_moving:
            motion_pos = self._track_moving_ball(frame, self.prev_frame, debug_frame, debug_info)
            if motion_pos:
                ball_pos = motion_pos  # Only update ball_pos if motion tracking found something
                self.consecutive_misses = 0  # Reset consecutive misses counter
            else:
                self.consecutive_misses += 1
                print(f"Missed detection for {self.consecutive_misses} consecutive frames")
                
                # If we've missed too many frames, reset to static detection
                if self.consecutive_misses >= self.max_consecutive_misses:
                    print("Too many consecutive misses - resetting to static detection")
                    # Store the completed trajectory and predictions before resetting
                    if len(self.trajectory) > 2:  # Only store if we have a meaningful trajectory
                        # First, create the completed trajectory including the last predicted position
                        self.completed_trajectory = self.trajectory.copy()
                        if hasattr(self, 'last_predicted_pos'):
                            print("Adding last predicted position to completed trajectory")
                            self.completed_trajectory.append(self.last_predicted_pos)
                            
                            # Now generate predictions using the complete trajectory
                            print("Generating final predictions using complete trajectory")
                            self.completed_predictions = self._predict_next_points(self.completed_trajectory)
                        else:
                            self.completed_predictions = self.prev_predicted_points.copy() if self.prev_predicted_points else []
                        
                        self.frames_since_completion = 0
                        print("Storing completed trajectory with predictions")
                    
                    self.is_moving = False
                    self.trajectory = []
                    self.predicted_points = []
                    self.prev_predicted_points = []
                    self.consecutive_misses = 0
                    self.frames_since_static = 0
                    # Add text to debug frame
                    cv2.putText(debug_frame, "RESET - Looking for static ball", 
                              (frame.shape[1]//2 - 150, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Update ball position and trajectory
        self.ball_pos = ball_pos
        if self.is_moving:
            if ball_pos:
                # If we have a real detection, clear predictions and add the new point
                if not self.trajectory or ball_pos != self.trajectory[-1]:
                    print(f"Adding new trajectory point: {ball_pos}")
                    self.trajectory.append(ball_pos)
                    self.predicted_points = []  # Clear predictions when we get a new detection
                    self.prev_predicted_points = []  # Also clear previous predictions
                    self.frames_since_prediction = 0  # Reset prediction display counter
                    self.frames_since_lost = 0  # Reset frames since lost counter
            else:
                # Increment frames since lost counter
                self.frames_since_lost += 1
                print(f"Frames since lost: {self.frames_since_lost}")
                
                # Generate predictions after delay if needed
                if self.frames_since_lost >= self.prediction_delay and len(self.trajectory) >= 2:
                    # Generate new predictions if we don't have any or if we have a new predicted search position
                    if (not self.predicted_points or 
                        (hasattr(self, 'last_predicted_pos') and 
                         self.last_predicted_pos not in self.trajectory)):
                        print("Generating new predictions")
                        # Create a temporary trajectory including the predicted search position
                        temp_trajectory = self.trajectory.copy()
                        if hasattr(self, 'last_predicted_pos'):
                            temp_trajectory.append(self.last_predicted_pos)
                        
                        self.predicted_points = self._predict_next_points(temp_trajectory)
                        print(f"Generated {len(self.predicted_points)} predicted points")
                        # Keep the predictions visible
                        self.prev_predicted_points = self.predicted_points
                        # Reset the prediction display counter when generating new predictions
                        self.frames_since_prediction = 0
                else:
                    print(f"Not generating predictions yet - need {self.prediction_delay} frames (currently at {self.frames_since_lost})")
        
        # Draw trajectory visualization
        if len(self.trajectory) > 1:
            # Draw active trajectory
            if self.is_moving:
                self.trajectory_tracer.draw_trajectory(
                    debug_frame,
                    self.trajectory,
                    self.prev_predicted_points,
                    color=(0, 255, 0),
                    thickness=2,
                    is_moving=True
                )
                
                # Draw on output frame
                self.trajectory_tracer.draw_trajectory(
                    self.output_frame,
                    self.trajectory,
                    self.prev_predicted_points,
                    color=(0, 255, 0),
                    thickness=2,
                    is_moving=True
                )
        
        # Draw completed trajectory if it exists
        if self.completed_trajectory:
            self.trajectory_tracer.draw_trajectory(
                debug_frame,
                self.completed_trajectory,
                self.completed_predictions,
                color=(0, 255, 0),
                thickness=2,
                is_moving=False
            )
            
            # Draw on output frame
            self.trajectory_tracer.draw_trajectory(
                self.output_frame,
                self.completed_trajectory,
                self.completed_predictions,
                color=(0, 255, 0),
                thickness=2,
                is_moving=False
            )
        
        # Store current frame for next iteration
        self.prev_frame = frame.copy()
        
        return {
            'ball_pos': self.ball_pos,
            'is_moving': self.is_moving,
            'trajectory': self.trajectory.copy(),
            'predicted_points': self.predicted_points.copy() if self.predicted_points else [],
            'prev_predicted_points': self.prev_predicted_points.copy() if self.prev_predicted_points else [],
            'debug_info': debug_info,
            'output_frame': self.output_frame  # Add clean output frame to return dict
        }

    def _track_moving_ball(self, current_frame: np.ndarray, prev_frame: np.ndarray, debug_frame: np.ndarray, debug_info: Dict) -> Optional[Tuple[int, int]]:
        """Track the moving ball using motion detection."""
        if prev_frame is None:
            return None
            
        # Get the last known position
        last_pos = self.last_known_static_pos if not self.trajectory else self.trajectory[-1]
        if not last_pos:
            return None
            
        x, y = last_pos
        height, width = current_frame.shape[:2]
        
        # If we have at least 2 points, predict next position
        predicted_pos = None
        if len(self.trajectory) >= 2:
            # Get last two points
            p1 = self.trajectory[-2]
            p2 = self.trajectory[-1]
            
            # Calculate velocity vector
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Add gravity factor - assume parabolic motion
            gravity_factor = 2  # pixels per frame^2
            dy += gravity_factor  # Ball will fall faster than linear prediction
            
            # Predict next position
            predicted_x = int(p2[0] + dx)
            predicted_y = int(p2[1] + dy)
            predicted_pos = (predicted_x, predicted_y)
            
            # Store this predicted position for trajectory calculations
            self.last_predicted_pos = predicted_pos
            
            # Draw prediction on debug frame
            cv2.circle(debug_frame, predicted_pos, 5, (0, 255, 255), -1)  # Yellow dot
            cv2.putText(debug_frame, "Predicted", 
                      (predicted_x + 10, predicted_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Define search areas - one around last known position and one around prediction if available
        search_areas = []
        
        # Always include a search area around last known position
        main_search = {
            'x_start': max(0, x - 50),
            'x_end': min(width, x + 50),
            'y_start': max(0, y - int(height * 0.3)),
            'y_end': min(height, y + 20),
            'is_predicted': False
        }
        search_areas.append(main_search)
        
        # Add smaller, more lenient search area around prediction if available
        if predicted_pos:
            px, py = predicted_pos
            pred_search = {
                'x_start': max(0, px - 40),  # Increased from 30 to 40
                'x_end': min(width, px + 40),
                'y_start': max(0, py - 40),
                'y_end': min(height, py + 40),
                'is_predicted': True
            }
            search_areas.append(pred_search)
        
        # Find the most likely ball candidate from motion
        best_pos = None
        max_score = 0
        best_motion_diff = None
        
        for search_area in search_areas:
            x_start = search_area['x_start']
            x_end = search_area['x_end']
            y_start = search_area['y_start']
            y_end = search_area['y_end']
            is_predicted = search_area['is_predicted']
            
            # Extract search regions
            roi_current = current_frame[y_start:y_end, x_start:x_end]
            roi_prev = prev_frame[y_start:y_end, x_start:x_end]
            
            if roi_current.size == 0 or roi_prev.size == 0:
                continue
            
            # Calculate motion
            diff = cv2.absdiff(roi_current, roi_prev)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            # More lenient motion threshold in predicted area
            threshold = 15 if is_predicted else 25
            _, motion_mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
            
            # Find contours of motion
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw search area
            color = (255, 255, 0) if is_predicted else (0, 0, 255)  # Yellow for predicted, red for main
            cv2.rectangle(debug_frame, (x_start, y_start), (x_end, y_end), color, 2)
            label = "Predicted Search Area" if is_predicted else "Motion Search Area"
            cv2.putText(debug_frame, label,
                      (x_start, max(20, y_start - 10)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Create motion visualization
            motion_diff = roi_current.copy()
            
            # Add motion mask overlay
            motion_overlay = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
            motion_overlay[..., 0] = 0
            motion_overlay[..., 2] = 0
            cv2.addWeighted(motion_diff, 1.0, motion_overlay, 0.5, 0, motion_diff)
            
            # Draw all contours first in blue
            cv2.drawContours(motion_diff, contours, -1, (255, 0, 0), 1)
            
            # Convert ROI to grayscale for brightness check
            roi_gray = cv2.cvtColor(roi_current, cv2.COLOR_BGR2GRAY)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                # More lenient size constraints for predicted area
                min_area = 3 if is_predicted else 10  # Reduced from 5 to 3
                max_area = 200 if is_predicted else 100  # Increased from 150 to 200
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                    
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw area for all contours
                cv2.putText(motion_diff, f"A:{area:.0f}", 
                          (cx, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Filter by area
                if area < min_area or area > max_area:
                    cv2.drawContours(motion_diff, [contour], -1, (0, 0, 255), 1)
                    if area < min_area:
                        cv2.putText(motion_diff, "too small", 
                                  (cx, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    else:
                        cv2.putText(motion_diff, "too big", 
                                  (cx, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    continue
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # More lenient circularity threshold for predicted area
                min_circularity = 0.2 if is_predicted else 0.5  # Reduced from 0.3 to 0.2
                if circularity < min_circularity:
                    cv2.drawContours(motion_diff, [contour], -1, (0, 255, 255), 1)
                    cv2.putText(motion_diff, f"not circular ({circularity:.2f})", 
                              (cx, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    continue
                
                # Convert to frame coordinates
                frame_x = x_start + cx
                frame_y = y_start + cy
                
                # Create mask for brightness check
                mask = np.zeros_like(roi_gray)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_brightness = cv2.mean(roi_gray, mask=mask)[0]
                
                # Score components
                distance = np.sqrt((frame_x - x)**2 + (frame_y - y)**2)
                height_score = (y - frame_y) / (height * 0.3)  # Higher score for moving up
                area_score = 1.0 - (area / max_area)
                brightness_score = mean_brightness / 255.0
                circularity_score = circularity
                distance_score = 1.0 - min(1.0, distance / 100)
                
                # Prediction score - if we have a predicted position, prefer candidates closer to it
                prediction_score = 0.0
                if predicted_pos:
                    pred_distance = np.sqrt((frame_x - predicted_pos[0])**2 + (frame_y - predicted_pos[1])**2)
                    prediction_score = 1.0 - min(1.0, pred_distance / 50)
                
                # Combined score with adjusted weights
                base_score = (height_score * 0.2 +
                            area_score * 0.1 +
                            brightness_score * 0.3 +
                            circularity_score * 0.2 +
                            distance_score * 0.2)
                
                # Boost score for candidates in predicted area
                if is_predicted:
                    base_score *= 1.5  # Increased from 1.2 to 1.5 (50% boost)
                
                # Add prediction score if available
                final_score = base_score + (prediction_score * 0.4 if predicted_pos else 0)  # Increased from 0.3 to 0.4
                
                # Draw valid candidate in green with detailed scores
                cv2.drawContours(motion_diff, [contour], -1, (0, 255, 0), 2)
                
                score_info = [
                    (f"Candidate {i+1} ({'Predicted' if is_predicted else 'Main'})", -55),
                    (f"Area: {area:.1f}px", -45),
                    (f"Bright: {brightness_score:.2f}", -35),
                    (f"Circ: {circularity:.2f}", -25),
                    (f"Height: {height_score:.2f}", -15),
                    (f"Dist: {distance_score:.2f}", -5),
                    (f"Pred: {prediction_score:.2f}" if predicted_pos else "", 5),
                    (f"Score: {final_score:.2f}", 15)
                ]
                
                for text, y_offset in score_info:
                    if text:  # Only draw non-empty strings
                        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        cv2.rectangle(motion_diff,
                                    (cx - text_w//2 - 2, cy + y_offset - text_h - 2),
                                    (cx + text_w//2 + 2, cy + y_offset + 2),
                                    (0, 0, 0), -1)
                        cv2.putText(motion_diff, text,
                                  (cx - text_w//2, cy + y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                if final_score > max_score:
                    max_score = final_score
                    best_pos = (frame_x, frame_y)
                    best_motion_diff = motion_diff.copy()
                    # Draw star or indicator for best candidate
                    cv2.drawContours(motion_diff, [contour], -1, (255, 255, 0), 3)
                    cv2.putText(motion_diff, "BEST CANDIDATE", 
                              (cx - 40, cy - 65),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)
        
        # Update motion visualization with the best view
        if best_motion_diff is not None:
            debug_info['motion_view'] = best_motion_diff
        
        return best_pos
    
    def reset(self):
        """Reset the tracker state."""
        self.ball_pos = None
        self.prev_frame = None
        self.trajectory = []
        self.predicted_points = []
        self.prev_predicted_points = []
        self.is_moving = False
        self.predicted_trajectory = []
        self.last_known_static_pos = None
        self.frames_since_static = 0
        self.max_static_loss_frames = 2
        self.consecutive_misses = 0
        self.completed_trajectory = []
        self.completed_predictions = []
        self.frames_since_completion = 0
        self.recent_motion_areas = [] 
        self.frames_since_lost = 0  # Reset frames since lost counter