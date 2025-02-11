import cv2
import numpy as np
from typing import Tuple
from ball_tracker import BallTracker
from trajectory_tracer import TrajectoryTracer
from tqdm import tqdm

class VideoProcessor:
    def __init__(self, ball_tracker: BallTracker, trajectory_tracer: TrajectoryTracer):
        self.ball_tracker = ball_tracker
        self.trajectory_tracer = trajectory_tracer
    
    def _get_video_properties(self, cap: cv2.VideoCapture) -> Tuple[int, int, int, float]:
        """Get video properties."""
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        return width, height, total_frames, fps
    
    def process_video(self, input_path: str, output_path: str):
        """Process the input video and generate output with ball tracking and trajectory."""
        # Open video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        
        # Get video properties
        width, height, total_frames, fps = self._get_video_properties(cap)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Reset tracker state
        self.ball_tracker.reset()
        
        # Process frames
        pbar = tqdm(total=total_frames, desc="Processing frames")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect ball in current frame
            ball_pos = self.ball_tracker.detect_ball(frame)
            
            # Get current trajectory
            trajectory = self.ball_tracker.get_trajectory()
            
            # Draw trajectory
            frame = self.trajectory_tracer.draw_trajectory(frame, trajectory)
            
            # Draw current ball position
            if ball_pos is not None:
                cv2.circle(frame, ball_pos, 5, (0, 255, 0), -1)
            
            # Write frame
            out.write(frame)
            pbar.update(1)
        
        # Clean up
        pbar.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows() 