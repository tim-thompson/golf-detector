import cv2
import numpy as np
import torch
import argparse
from ball_tracker import BallTracker
from trajectory_tracer import TrajectoryTracer
from video_processor import VideoProcessor

def parse_args():
    parser = argparse.ArgumentParser(description='Golf Ball Tracking Application')
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output video file')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU acceleration if available')
    parser.add_argument('--trace-color', type=str, default='white', help='Color of the trajectory trace')
    parser.add_argument('--trace-thickness', type=int, default=2, help='Thickness of the trajectory trace')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize components
    ball_tracker = BallTracker(use_gpu=args.use_gpu)
    trajectory_tracer = TrajectoryTracer(
        color=args.trace_color,
        thickness=args.trace_thickness
    )
    video_processor = VideoProcessor(
        ball_tracker=ball_tracker,
        trajectory_tracer=trajectory_tracer
    )
    
    # Process the video
    video_processor.process_video(args.input, args.output)

if __name__ == '__main__':
    main() 