import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from tqdm import tqdm
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
        self.master.title("JoJo Face Stylization")
        self.master.geometry("1400x800")
        self.master.configure(bg="#708090")  # Set background color

        # Main title for the application
        self.title_label = Label(master, text="🎨 JoJo Face Stylization 🎨", 
                                 font=("Helvetica", 20, "bold"), fg="#ffffff", bg="#708090")
        self.title_label.pack(pady=20)

        # Frame for video display
        self.video_frame = Frame(master, bg="#ffffff", bd=2, relief="groove")
        self.video_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Labels for original and stylized images
        self.original_label = Label(self.video_frame, text="Original Version", font=("Arial", 14, "bold"), 
                                    fg="#0073e6", bg="#ffffff")
        self.original_label.grid(row=0, column=0, padx=10, pady=10)

        self.stylized_label = Label(self.video_frame, text="Anime Version", font=("Arial", 14, "bold"), 
                                    fg="#ff6600", bg="#ffffff")
        self.stylized_label.grid(row=0, column=1, padx=10, pady=10)

        # Labels to display images
        self.original_image_label = Label(self.video_frame, bg="#f0f0f0", bd=1, relief="solid")
        self.original_image_label.grid(row=1, column=0, padx=10, pady=10)

        self.stylized_image_label = Label(self.video_frame, bg="#f0f0f0", bd=1, relief="solid")
        self.stylized_image_label.grid(row=1, column=1, padx=10, pady=10)


        # Frame for buttons
        self.button_frame = Frame(master, bg="#708090")
        self.button_frame.pack(pady=20)

        self.upload_button = Button(self.button_frame, text="Upload Image 🖼️", command=self.upload_image, 
                                    font=("Helvetica", 12), bg="#cce5ff", activebackground="#99ccff", bd=2)
        self.upload_button.grid(row=0, column=0, padx=20, ipadx=10)

        self.start_button = Button(self.button_frame, text="Start Real-time 🎥", command=self.start_stylization, 
                                   font=("Helvetica", 12), bg="#d1f0c0", activebackground="#b8e994", bd=2)
        self.start_button.grid(row=0, column=1, padx=20, ipadx=10)

        self.stop_button = Button(self.button_frame, text="Stop Real-time ✋", command=self.stop_stylization, 
                                  font=("Helvetica", 12), bg="#ffcccc", activebackground="#ff9999", bd=2)
        self.stop_button.grid(row=0, column=2, padx=20, ipadx=10)

        # Initializing the model
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
            self.original_image_label.config(image='')
            self.stylized_image_label.config(image='')

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return

        # Load and display the original image
        raw_image = cv2.imread(file_path)
        frame_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        self.display_image(frame_rgb, self.original_image_label)

        # Stylize the image
        stylized_output = self.stylize_image(frame_rgb)

        # Display the stylized image
        self.display_image(stylized_output, self.stylized_image_label)

    def stylize_image(self, frame_rgb):
        original_height, original_width = frame_rgb.shape[:2]

    # Resize image to 256x256 for model input
        resized_image = cv2.resize(frame_rgb, (256, 256))
        image = resized_image / 127.5 - 1  # Normalize image to range [-1, 1]
        image = image.transpose(2, 0, 1)  # Change shape to [C, H, W]
        image = torch.tensor(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = self.model(image.float())  # Forward pass through model

    # Process the output and convert it back to an image
            output = output.squeeze(0).detach().numpy()  # Remove batch dimension
            output = output.transpose(1, 2, 0)  # Convert shape to [H, W, C]
            output = (output + 1) * 127.5  # Denormalize image to range [0, 255]
            output = np.clip(output, 0, 255).astype(np.uint8)  # Clip to valid pixel range
    # Resize stylized output back to original image size
            stylized_output = cv2.resize(output, (original_width, original_height))
        return stylized_output


    def display_image(self, image_array, label, width=640, height=480):
        # Resize image_array to fixed dimensions
        resized_image = cv2.resize(image_array, (width, height))
        image = Image.fromarray(resized_image)
        image_tk = ImageTk.PhotoImage(image)
        label.imgtk = image_tk
        label.configure(image=image_tk)

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stylized_output = self.stylize_image(frame_rgb)
                self.display_image(frame_rgb, self.original_image_label)
                self.display_image(stylized_output, self.stylized_image_label)
            self.master.after(10, self.update_frame)

    def on_closing(self):
        self.stop_stylization()
        self.master.destroy()



if __name__ == '__main__':
    root = Tk()
    app = JoJoFaceStylizationApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
