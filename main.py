import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import torch
from PIL import Image
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

class YOLOv5Transformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to PIL for YOLOv5 compatibility
        img = Image.fromarray(img)
        
        # Perform inference
        results = model(img)
        
        # Convert results to image
        results.render()  # Renders predictions on the input image
        data = np.asarray(results.imgs[0])
        
        # Convert numpy array back to AVFrame
        new_frame = av.VideoFrame.from_ndarray(data, format="bgr24")
        return new_frame

st.title("YOLOv5 Real-Time Object Detection")
st.write("Detect objects using YOLOv5 in real-time from your webcam")

# Use webrtc_streamer to access webcam and display video
webrtc_streamer(key="example", video_processor_factory=YOLOv5Transformer)
