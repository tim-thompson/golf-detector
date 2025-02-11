import cv2
import numpy as np
from typing import List, Tuple
from scipy.interpolate import splprep, splev

class TrajectoryTracer:
    def __init__(self, color: str = 'white', thickness: int = 2):
        self.thickness = thickness
        self.color = self._parse_color(color)
        self.min_points_for_smoothing = 4
        
    def _parse_color(self, color: str) -> Tuple[int, int, int]:
        """Convert color name to BGR tuple."""
        color_map = {
            'white': (255, 255, 255),
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255)
        }
        return color_map.get(color.lower(), (255, 255, 255))
    
    def _smooth_trajectory(self, points: List[Tuple[int, int]], smoothing_factor: float = 0.3) -> np.ndarray:
        """Apply spline interpolation to smooth the trajectory."""
        if len(points) < self.min_points_for_smoothing:
            return np.array(points)
        
        # Separate x and y coordinates
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        
        # Fit spline
        tck, u = splprep([x, y], s=smoothing_factor, k=3)
        
        # Generate more points for smooth curve
        u_new = np.linspace(0, 1, len(points) * 5)
        smooth_points = splev(u_new, tck)
        
        return np.column_stack((smooth_points[0], smooth_points[1])).astype(np.int32)
    
    def draw_trajectory(self, frame: np.ndarray, trajectory: List[Tuple[int, int]]) -> np.ndarray:
        """Draw the ball trajectory on the frame with anti-aliasing and smooth curves."""
        if len(trajectory) < 2:
            return frame
        
        # Create a separate layer for the trajectory
        overlay = frame.copy()
        
        # Smooth the trajectory
        smooth_trajectory = self._smooth_trajectory(trajectory)
        
        # Draw the smooth curve with anti-aliasing
        cv2.polylines(
            overlay,
            [smooth_trajectory],
            False,
            self.color,
            thickness=self.thickness,
            lineType=cv2.LINE_AA
        )
        
        # Add fade effect for older parts of the trajectory
        if len(smooth_trajectory) > 1:
            for i in range(len(smooth_trajectory) - 1):
                alpha = (i + 1) / len(smooth_trajectory)
                pt1 = tuple(smooth_trajectory[i])
                pt2 = tuple(smooth_trajectory[i + 1])
                cv2.line(
                    overlay,
                    pt1,
                    pt2,
                    self.color,
                    thickness=max(1, int(self.thickness * alpha)),
                    lineType=cv2.LINE_AA
                )
        
        # Blend the trajectory with the original frame
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame 