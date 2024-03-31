import os
import cv2

def natural_sort_key(s):
    """Key function for natural sorting of strings."""
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def create_video_from_frames(frames_dir, output_video_path, fps=30):
    # Get list of frames and sort them using natural sorting
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')], key=natural_sort_key)

    # Get frame dimensions from first frame
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape

    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec (use 'XVID' for AVI format)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Iterate over frames and add them to video
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    # Release VideoWriter object
    video_writer.release()

# Example usage:
frames_dir = 'deadlift\d_2\save_dir'
output_video_path = 'deadlift\d_pose_2.mp4'
create_video_from_frames(frames_dir, output_video_path)
