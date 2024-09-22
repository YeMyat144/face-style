import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from tkinter import Tk, Button, Label, Frame
from PIL import Image, ImageTk
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, num_channel):
        super(ResBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(num_channel, num_channel, 3, 1, 1),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channel, num_channel, 3, 1, 1),
            nn.BatchNorm2d(num_channel))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs):
        output = self.conv_layer(inputs)
        output = self.activation(output + inputs)
        return output

class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 2, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

    def forward(self, inputs):
        output = self.conv_layer(inputs)
        return output

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_last=False):
        super(UpBlock, self).__init__()
        self.is_last = is_last
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1))
        self.act = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))
        self.last_act = nn.Tanh()

    def forward(self, inputs):
        output = self.conv_layer(inputs)
        if self.is_last:
            output = self.last_act(output)
        else:
            output = self.act(output)
        return output

class SimpleGenerator(nn.Module):
    def __init__(self, num_channel=32, num_blocks=4):
        super(SimpleGenerator, self).__init__()
        self.down1 = DownBlock(3, num_channel)
        self.down2 = DownBlock(num_channel, num_channel * 2)
        self.down3 = DownBlock(num_channel * 2, num_channel * 3)
        self.down4 = DownBlock(num_channel * 3, num_channel * 4)
        res_blocks = [ResBlock(num_channel * 4)] * num_blocks
        self.res_blocks = nn.Sequential(*res_blocks)
        self.up1 = UpBlock(num_channel * 4, num_channel * 3)
        self.up2 = UpBlock(num_channel * 3, num_channel * 2)
        self.up3 = UpBlock(num_channel * 2, num_channel)
        self.up4 = UpBlock(num_channel, 3, is_last=True)

    def forward(self, inputs):
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down4 = self.res_blocks(down4)
        up1 = self.up1(down4)
        up2 = self.up2(up1 + down3)
        up3 = self.up3(up2 + down2)
        up4 = self.up4(up3 + down1)
        return up4

class JoJoFaceStylizationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Real-time JoJo Face Stylization")
        self.master.geometry("1000x600")

        self.video_frame = Frame(master)
        self.video_frame.pack()

        self.video_label = Label(self.video_frame)
        self.video_label.pack()

        self.start_button = Button(master, text="Start", command=self.start_stylization)
        self.start_button.pack(side="left", padx=10, pady=20)

        self.stop_button = Button(master, text="Stop", command=self.stop_stylization)
        self.stop_button.pack(side="right", padx=10, pady=20)

        self.cap = None
        self.model = self.load_model()
        if self.model:
            print("Model loaded successfully!")

    def load_model(self):
        model = SimpleGenerator()
        if not os.path.exists('weight.pth'):
            print("Error: weight.pth file not found.")
            return None
        model.load_state_dict(torch.load('weight.pth', map_location='cpu'))
        model.eval()
        return model

    def start_stylization(self):
        # Open camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return
        self.update_frame()

    def stop_stylization(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            self.video_label.config(image='')

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Convert from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize for model input
                raw_image = cv2.resize(frame_rgb, (256, 256))
                image = raw_image / 127.5 - 1  # Normalize image to range [-1, 1]
                image = image.transpose(2, 0, 1)  # Change shape to [C, H, W]
                image = torch.tensor(image).unsqueeze(0)  # Add batch dimension

                with torch.no_grad():
                    output = self.model(image.float())  # Forward pass through model

                output = output.squeeze(0).detach().numpy()  # Remove batch dimension
                output = output.transpose(1, 2, 0)  # Convert shape to [H, W, C]
                output = (output + 1) * 127.5  # Denormalize image to range [0, 255]
                output = np.clip(output, 0, 255).astype(np.uint8)  # Clip to valid pixel range

                # Resize output to original frame size
                output = cv2.resize(output, (frame.shape[1], frame.shape[0]))

                # Combine original frame and stylized output side-by-side
                combined_output = np.hstack((frame_rgb, output))  # Use frame_rgb here

                # Convert to ImageTk for displaying in Tkinter
                combined_output = Image.fromarray(combined_output)
                combined_output = ImageTk.PhotoImage(combined_output)

                self.video_label.imgtk = combined_output
                self.video_label.configure(image=combined_output)
            
            # Repeat every 10 milliseconds
            self.master.after(10, self.update_frame)

    def on_closing(self):
        self.stop_stylization()
        self.master.destroy()

if __name__ == '__main__':
    root = Tk()
    app = JoJoFaceStylizationApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
