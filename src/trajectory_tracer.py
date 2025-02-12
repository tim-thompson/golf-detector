import cv2
import numpy as np
from typing import List, Tuple, Union
from scipy.interpolate import splprep, splev

class TrajectoryTracer:
    def __init__(self):
        self.animation_progress = 0.0
        self.is_animating = False
        self.smoothed_points = None
        self.display_frames = 150  # Display for 5 seconds at 30fps
        self.fade_start_frame = 120  # Start fading after 4 seconds
        self.frames_since_completion = 0
        self.is_completed = False
    
    def _smooth_trajectory(self, points: List[Tuple[int, int]], smoothing_factor: float = 0.5) -> List[Tuple[int, int]]:
        """Create a smoothed trajectory using spline interpolation."""
        if len(points) < 4:
            return points
            
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        
        # Parameterize points by their cumulative distance
        t = np.zeros(len(points))
        for i in range(1, len(points)):
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
            t[i] = t[i-1] + np.sqrt(dx*dx + dy*dy)
        
        if t[-1] < 1e-6:
            return points
            
        # Normalize parameter values
        t = t / t[-1]
        
        try:
            # Fit spline with error handling
            tck, _ = splprep([x, y], u=t, s=smoothing_factor * len(points), k=3)
            
            # Generate more points along spline for smoother appearance
            u_new = np.linspace(0, 1, num=max(300, len(points) * 8))
            x_smooth, y_smooth = splev(u_new, tck)
            
            # Convert to list of points
            return list(zip(x_smooth.astype(int), y_smooth.astype(int)))
            
        except Exception as e:
            print(f"Spline interpolation failed: {str(e)}")
            return points
    
    def _calculate_animation_speed(self, progress: float, points: List[Tuple[int, int]]) -> float:
        """Calculate animation speed based on position in trajectory."""
        if len(points) < 2:
            return 0.04  # Default speed
            
        # Find the highest point (apex) of the trajectory
        y_coords = [p[1] for p in points]
        min_y_idx = y_coords.index(min(y_coords))  # Remember: y is inverted in image coordinates
        apex_progress = min_y_idx / len(points)
        
        # Calculate distance from apex (as a percentage of total trajectory)
        distance_from_apex = abs(progress - apex_progress)
        
        # Wider apex influence (30% of trajectory) and smoother transition
        apex_influence = max(0.0, 1.0 - (distance_from_apex / 0.3))
        apex_influence = apex_influence ** 0.5  # Square root for smoother transition
        
        # Base speeds for different phases
        start_speed = 0.1    # Very fast at start
        apex_speed = 0.01    # Very slow at apex
        end_speed = 0.06     # Medium-fast at end
        
        if progress < apex_progress:
            # Initial portion: Start very fast, smoothly slow down approaching apex
            t = progress / apex_progress
            # Cubic easing for smoother deceleration
            t = t * t * (3 - 2 * t)
            speed = start_speed * (1.0 - t) + apex_speed * t
        else:
            # After apex: Maintain slow speed briefly, then gradually speed up
            t = (progress - apex_progress) / (1.0 - apex_progress)
            # Delayed acceleration using cubic easing
            t = t * t * t
            speed = apex_speed + (end_speed - apex_speed) * t
        
        # Apply apex influence with smoother transition
        final_speed = speed * (1.0 - 0.8 * apex_influence)
        
        # Ensure minimum speed to prevent stalling
        return max(0.008, final_speed)
    
    def draw_trajectory(self, frame: np.ndarray, real_points: List[Tuple[int, int]], 
                       predicted_points: List[Tuple[int, int]] = None,
                       color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2,
                       is_moving: bool = False) -> None:
        """Draw the trajectory with smooth animation."""
        # Combine all points
        points = real_points + (predicted_points if predicted_points else [])
        if not points:
            return
            
        # Generate or update smoothed points
        if not self.smoothed_points or len(points) != len(self.smoothed_points):
            self.smoothed_points = self._smooth_trajectory(points)
            # Only start new animation if we don't have one running
            if not self.is_animating:
                self.is_animating = True
                self.animation_progress = 0.0
                self.is_completed = False
                self.frames_since_completion = 0
        
        # Handle animation
        if self.is_animating:
            # Update animation progress
            if not self.is_completed:
                # Calculate physics-based animation speed
                speed = self._calculate_animation_speed(self.animation_progress, points)
                self.animation_progress = min(1.0, self.animation_progress + speed)
                
                if self.animation_progress >= 1.0:
                    self.is_completed = True
        
        # Handle fade out
        alpha = 1.0
        if self.is_completed:
            self.frames_since_completion += 1
            if self.frames_since_completion >= self.fade_start_frame:
                fade_progress = (self.frames_since_completion - self.fade_start_frame) / (self.display_frames - self.fade_start_frame)
                alpha = max(0.0, 1.0 - fade_progress)
            
            if self.frames_since_completion >= self.display_frames:
                self.is_animating = False
                self.animation_progress = 0.0
                self.frames_since_completion = 0
                self.is_completed = False
                return
        
        # Calculate visible portion of trajectory
        num_points = max(2, int(len(self.smoothed_points) * self.animation_progress))
        visible_points = self.smoothed_points[:num_points]
        
        # Create overlay for anti-aliased drawing
        overlay = frame.copy()
        
        # Draw the trajectory with a black border for better visibility
        pts_array = np.array(visible_points, np.int32).reshape((-1, 1, 2))
        
        # Draw black border
        cv2.polylines(overlay, [pts_array], False, (0, 0, 0), thickness + 4, cv2.LINE_AA)
        
        # Draw main colored line with alpha
        color_with_alpha = (
            int(color[0] * alpha),
            int(color[1] * alpha),
            int(color[2] * alpha)
        )
        cv2.polylines(overlay, [pts_array], False, color_with_alpha, thickness, cv2.LINE_AA)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
    def draw_trajectory_on_frame(self, frame: np.ndarray, trajectory: List[Tuple[int, int]], 
                               color: Union[str, Tuple[int, int, int]] = (255, 255, 255),
                               thickness: int = 2,
                               alpha: float = 0.7,
                               line_style: int = cv2.LINE_AA) -> np.ndarray:
        """
        Draw the ball trajectory on the frame.
        
        Args:
            frame: Input frame
            trajectory: List of trajectory points
            color: Color in BGR format or color name string
            thickness: Line thickness
            alpha: Transparency of the trajectory (0-1)
            line_style: OpenCV line style or 'dashed' for dashed lines
        """
        if len(trajectory) < 2:
            return frame
        
        # Handle string color names
        if isinstance(color, str):
            color_map = {
                'white': (255, 255, 255),
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'blue': (255, 0, 0),
                'yellow': (0, 255, 255)
            }
            color = color_map.get(color.lower(), (255, 255, 255))
        
        # Create a separate layer for the trajectory
        overlay = frame.copy()
        
        # Smooth the trajectory if we have enough points
        if len(trajectory) >= 4:
            x_smooth, y_smooth = self._smooth_trajectory(trajectory)
        else:
            x_smooth = np.array([p[0] for p in trajectory])
            y_smooth = np.array([p[1] for p in trajectory])
        
        # Draw the smooth curve
        if line_style == 'dashed':
            # Create dashed line effect by drawing segments
            dash_length = 10  # Length of each dash
            gap_length = 5    # Length of gap between dashes
            
            for i in range(0, len(x_smooth) - 1):
                pt1 = (int(x_smooth[i]), int(y_smooth[i]))
                pt2 = (int(x_smooth[i + 1]), int(y_smooth[i + 1]))
                
                # Calculate the vector between points
                dx = pt2[0] - pt1[0]
                dy = pt2[1] - pt1[1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < 1:  # Skip if points are too close
                    continue
                
                # Normalize the vector
                dx /= dist
                dy /= dist
                
                # Draw dashed segments
                pos = 0
                while pos < dist:
                    # Calculate segment start/end
                    start_x = int(pt1[0] + dx * pos)
                    start_y = int(pt1[1] + dy * pos)
                    end_pos = min(pos + dash_length, dist)
                    end_x = int(pt1[0] + dx * end_pos)
                    end_y = int(pt1[1] + dy * end_pos)
                    
                    # Draw the dash
                    cv2.line(overlay,
                            (start_x, start_y),
                            (end_x, end_y),
                            color,
                            thickness=thickness,
                            lineType=cv2.LINE_AA)
                    
                    # Move to next segment
                    pos += dash_length + gap_length
        else:
            # Draw solid line
            cv2.polylines(
                overlay,
                [np.column_stack((x_smooth.astype(int), y_smooth.astype(int))).tolist()],
                False,
                color,
                thickness=thickness,
                lineType=line_style
            )
            
            # Add fade effect for older parts of the trajectory
            if len(x_smooth) > 1:
                for i in range(len(x_smooth) - 1):
                    pt_alpha = (i + 1) / len(x_smooth)
                    pt1 = (int(x_smooth[i]), int(y_smooth[i]))
                    pt2 = (int(x_smooth[i + 1]), int(y_smooth[i + 1]))
                    cv2.line(
                        overlay,
                        pt1,
                        pt2,
                        color,
                        thickness=max(1, int(thickness * pt_alpha)),
                        lineType=line_style
                    )
        
        # Blend the trajectory with the original frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame 