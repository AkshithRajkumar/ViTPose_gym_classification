import gradio as gr
import os
import subprocess

# Define function to run inference1.py
def run_inference(video_file):
    # Run inference1.py on the uploaded video
    output_dir = "processed_frames"
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(["python", "inference1.py", "--video", video_file.name, "--output", output_dir])

    # Get list of processed frames
    processed_frames = [os.path.join(output_dir, frame) for frame in os.listdir(output_dir)]

    return processed_frames

# Define function to run test_classifier.py
def classify_video(processed_frames):
    # Run test_classifier.py on the processed frames
    classification_output = subprocess.check_output(["python", "test_classifier.py", "--frames", processed_frames])
    
    return classification_output.decode("utf-8")

# Create Gradio interface
input_video = gr.inputs.Video(label="Upload Video")
processed_video = gr.outputs.Video(label="Processed Video")
classification_output = gr.outputs.Textbox(label="Classification Output")

gr.Interface(
    [run_inference, classify_video],
    inputs=input_video,
    outputs=[processed_video, classification_output],
    title="Video Processing and Classification",
    description="Upload a video to process and classify."
).launch()
