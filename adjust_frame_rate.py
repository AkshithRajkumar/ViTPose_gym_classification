import cv2

# Function to adjust the number of frames in a video to exactly 120 frames
def adjust_frame_count(input_video_path, output_video_path):
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    # Get total number of frames in the input video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame skip factor to achieve exactly 120 frames
    skip_factor = total_frames / 120

    # Initialize variables
    frames = []
    frame_number = 0

    # Read frames and select every 'skip_factor'-th frame
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if int(frame_number) % skip_factor == 0:
                frames.append(frame)
            frame_number += 1
        else:
            break

    # Release video capture object
    cap.release()

    # Write the selected frames to the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()

# Example usage
input_video_path = "deadlift\d_30.mp4"
output_video_path = "deadlift\d_30_output.mp4"

# Adjust the number of frames in the video to exactly 120 frames
adjust_frame_count(input_video_path, output_video_path)
