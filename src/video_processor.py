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
    
    def _create_debug_view(self, tracking_info: dict, frame_size: Tuple[int, int]) -> np.ndarray:
        """Create a multi-panel debug view."""
        height, width = frame_size
        debug_info = tracking_info['debug_info']
        
        # Create a more compact layout
        # Main view takes up 2/3 of the width on the left
        main_width = (width * 3) // 3 * 2
        main_height = height * 2
        
        # Debug panels on the right take up 1/3 width
        debug_panel_width = (width * 3) - main_width
        debug_panel_height = main_height // 5  # 5 equal height panels
        
        # Create canvas
        debug_canvas = np.zeros((main_height, width * 3, 3), dtype=np.uint8)
        
        # Function to resize maintaining aspect ratio
        def resize_and_pad(img, target_size, pad_color=(0,0,0)):
            if img is None:
                # Create a black panel with "No Data" text
                panel = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                cv2.putText(panel, "No Data", 
                           (target_size[0]//3, target_size[1]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                return panel
            
            h, w = img.shape[:2]
            scale = min(target_size[0]/w, target_size[1]/h)
            new_size = (int(w*scale), int(h*scale))
            
            resized = cv2.resize(img, new_size)
            
            padded = np.full((target_size[1], target_size[0], 3), pad_color, dtype=np.uint8)
            y_offset = (target_size[1] - new_size[1]) // 2
            x_offset = (target_size[0] - new_size[0]) // 2
            padded[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized
            
            return padded
        
        # Place main view (large, left side)
        main_view = resize_and_pad(debug_info['main_view'], (main_width, main_height))
        debug_canvas[0:main_height, 0:main_width] = main_view
        
        # Add vertical separator
        cv2.line(debug_canvas, (main_width, 0), (main_width, main_height), (100, 100, 100), 2)
        
        # Place debug panels on the right side
        debug_panels = [
            ('Search Area', debug_info['grayscale_view']),
            ('Threshold', debug_info['threshold_view']),
            ('Candidates', debug_info['candidates_view']),
            ('Motion', debug_info['motion_view'])
        ]
        
        for i, (label, panel) in enumerate(debug_panels):
            # Calculate panel position
            y_start = i * debug_panel_height
            
            # Create panel background
            panel_bg = np.zeros((debug_panel_height, debug_panel_width, 3), dtype=np.uint8)
            debug_canvas[y_start:y_start+debug_panel_height, main_width:] = panel_bg
            
            # Add panel label with dark background
            label_bg = np.zeros((30, debug_panel_width, 3), dtype=np.uint8)
            debug_canvas[y_start:y_start+30, main_width:] = label_bg
            
            # Add label text
            cv2.putText(debug_canvas, label, 
                       (main_width + 10, y_start + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Resize and place panel
            if panel is not None:
                resized_panel = resize_and_pad(panel, (debug_panel_width, debug_panel_height-30))
                debug_canvas[y_start+30:y_start+debug_panel_height, main_width:] = resized_panel
        
        # Status panel at the bottom
        status_y_start = 4 * debug_panel_height
        status_panel = np.zeros((debug_panel_height, debug_panel_width, 3), dtype=np.uint8)
        
        # Add status information
        status_text = [
            "Status:",
            f"Ball Moving: {tracking_info['is_moving']}",
            f"Ball Position: {tracking_info['ball_pos']}",
        ]
        
        for i, text in enumerate(status_text):
            y_pos = 30 + i * 30
            cv2.putText(status_panel, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Place status panel
        debug_canvas[status_y_start:status_y_start+debug_panel_height, main_width:] = status_panel
        
        # Add horizontal separators between debug panels
        for i in range(1, 5):
            y = i * debug_panel_height
            cv2.line(debug_canvas, (main_width, y), (width * 3, y), (100, 100, 100), 1)
        
        return debug_canvas
    
    def process_video(self, input_path: str, output_path: str, debug_output_path: str):
        """Process the input video and generate both main and debug output videos."""
        # Open video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")
        
        # Get video properties
        width, height, total_frames, fps = self._get_video_properties(cap)
        
        # Create video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        debug_out = cv2.VideoWriter(debug_output_path, fourcc, fps, (width * 3, height * 2))
        
        # Reset tracker state
        self.ball_tracker.reset()
        
        # Process frames
        pbar = tqdm(total=total_frames, desc="Processing frames")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Track ball in current frame
            tracking_info = self.ball_tracker.track_frame(frame)
            
            # Create main output frame
            output_frame = frame.copy()
            
            # Draw trajectory and predicted trajectory
            if tracking_info['trajectory']:
                # Draw active trajectory
                if tracking_info['is_moving']:
                    self.trajectory_tracer.draw_trajectory(
                        output_frame,
                        tracking_info['trajectory'],
                        tracking_info['prev_predicted_points'],
                        color=(0, 255, 0),
                        thickness=2,
                        is_moving=True
                    )
            
            # Draw completed trajectory if it exists
            if self.ball_tracker.completed_trajectory:
                self.trajectory_tracer.draw_trajectory(
                    output_frame,
                    self.ball_tracker.completed_trajectory,
                    self.ball_tracker.completed_predictions,
                    color=(0, 255, 0),
                    thickness=2,
                    is_moving=False
                )
            
            # Draw current ball position
            if tracking_info['ball_pos']:
                cv2.circle(output_frame, tracking_info['ball_pos'], 5, 
                          (0, 0, 255) if tracking_info['is_moving'] else (255, 0, 0), -1)
            
            # Create debug view
            debug_view = self._create_debug_view(tracking_info, (height, width))
            
            # Write frames
            out.write(output_frame)
            debug_out.write(debug_view)
            pbar.update(1)
        
        # Clean up
        pbar.close()
        cap.release()
        out.release()
        debug_out.release()
        cv2.destroyAllWindows() 