import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import numpy as np
import cv2
from torchvision.transforms.functional import to_tensor

class ObjectDetectionApp:
				def __init__(self, root):
								self.root = root
								self.root.title("YOLOv5 Object Detection with Tkinter")

								# Load YOLOv5 model
								self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

								# Setup UI
								self.setup_ui()

				def setup_ui(self):
								self.upload_btn = tk.Button(self.root, text='Upload Image', command=self.upload_image)
								self.upload_btn.pack(pady=20)

								self.image_label = tk.Label(self.root)
								self.image_label.pack(pady=20)

				def upload_image(self):
								file_path = filedialog.askopenfilename()
								if file_path:
												self.detect_objects(file_path)

				def detect_objects(self, image_path):
								img = Image.open(image_path)
								img_tensor = to_tensor(img).unsqueeze_(0)
								results = self.model(img_tensor)

								# Render detections
								results.render()
								for img in results.imgs:
												img = Image.fromarray(img.astype(np.uint8))
												self.display_image(img)

				def display_image(self, img):
								img_tk = ImageTk.PhotoImage(img)
								self.image_label.config(image=img_tk)
								self.image_label.image = img_tk  # Keep a reference!

if __name__ == '__main__':
				root = tk.Tk()
				app = ObjectDetectionApp(root)
				root.mainloop()
