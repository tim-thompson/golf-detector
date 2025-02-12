import cv2
import numpy as np
from typing import List, Tuple, Union
from scipy.interpolate import splprep, splev

class TrajectoryTracer:
    def __init__(self):
        self.min_points_for_smoothing = 4
    
    def _smooth_trajectory(self, points: List[Tuple[int, int]], smoothing_factor: float = 0.3) -> np.ndarray:
        """Apply spline interpolation to smooth the trajectory."""
        if len(points) < self.min_points_for_smoothing:
            return np.array(points)
        
        # Separate x and y coordinates
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        
        try:
            # Try to fit spline
            tck, u = splprep([x, y], s=smoothing_factor, k=min(3, len(points) - 1))
            
            # Generate more points for smooth curve
            u_new = np.linspace(0, 1, len(points) * 5)
            smooth_points = splev(u_new, tck)
            
            return np.column_stack((smooth_points[0], smooth_points[1])).astype(np.int32)
        except ValueError:
            # If spline fitting fails, return original points
            print(f"Smoothing failed for {len(points)} points, using original trajectory")
            return np.array(points)
    
    def draw_trajectory(self, frame: np.ndarray, trajectory: List[Tuple[int, int]], 
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
        if len(trajectory) >= self.min_points_for_smoothing:
            smooth_trajectory = self._smooth_trajectory(trajectory)
        else:
            smooth_trajectory = np.array(trajectory)
        
        # Draw the smooth curve
        if line_style == 'dashed':
            # Create dashed line effect by drawing segments
            dash_length = 10  # Length of each dash
            gap_length = 5    # Length of gap between dashes
            
            for i in range(0, len(smooth_trajectory) - 1):
                pt1 = tuple(smooth_trajectory[i])
                pt2 = tuple(smooth_trajectory[i + 1])
                
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
                [smooth_trajectory],
                False,
                color,
                thickness=thickness,
                lineType=line_style
            )
            
            # Add fade effect for older parts of the trajectory
            if len(smooth_trajectory) > 1:
                for i in range(len(smooth_trajectory) - 1):
                    pt_alpha = (i + 1) / len(smooth_trajectory)
                    pt1 = tuple(smooth_trajectory[i])
                    pt2 = tuple(smooth_trajectory[i + 1])
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