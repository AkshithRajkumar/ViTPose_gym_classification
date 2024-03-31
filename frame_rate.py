import cv2

# Input and output video file paths
input_video_path = "deadlift\d_6.mp4"
output_video_path = "deadlift\d_6_output.mp4"

# Open the input video file
cap = cv2.VideoCapture(input_video_path)

# Get the current frame rate of the input video
current_frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Define the desired frame rate for the output video
desired_frame_rate = 30  # Change this to the desired frame rate

# Calculate the ratio to adjust frame rate
frame_rate_ratio = desired_frame_rate / current_frame_rate

# Initialize variables
frame_count = 0
frame_number = 0

# Create VideoWriter object to write the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, desired_frame_rate, (int(cap.get(3)), int(cap.get(4))))

# Read frames from the input video and write to the output video
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame_count += 1
        if frame_count >= frame_rate_ratio:
            frame_count = 0
            out.write(frame)
            frame_number += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved as {output_video_path}")
