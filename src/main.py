import cv2
import numpy as np
import argparse
from ball_tracker import BallTracker
from trajectory_tracer import TrajectoryTracer
from video_processor import VideoProcessor

def parse_args():
    parser = argparse.ArgumentParser(description='Golf Ball Tracking Application')
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output video file')
    parser.add_argument('--debug-output', type=str, default='debug_output.mp4', help='Path to debug output video file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize components
    ball_tracker = BallTracker()
    trajectory_tracer = TrajectoryTracer()
    video_processor = VideoProcessor(
        ball_tracker=ball_tracker,
        trajectory_tracer=trajectory_tracer
    )
    
    # Process the video
    video_processor.process_video(
        input_path=args.input,
        output_path=args.output,
        debug_output_path=args.debug_output
    )

if __name__ == '__main__':
    main() 