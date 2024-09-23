import os
import cv2
import torch
import asyncio
import websockets
import numpy as np
from PIL import Image
from io import BytesIO
import base64
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
    
class RealTimeJoJoStylization:
    def __init__(self):
        self.model = self.load_model()
        if self.model:
            print("Model loaded successfully!")

    def load_model(self):
        model = SimpleGenerator()
        weight_path = '/current/weight.pth'
        if not os.path.exists(weight_path):
            print(f"Error: {weight_path} file not found.")
            return None
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        model.eval()
        return model

    async def process_frame(self, websocket, path):
        async for message in websocket:
            # Receive frame from client
            frame_data = base64.b64decode(message)
            image = Image.open(BytesIO(frame_data))
            image = np.array(image)

            # Convert image from RGB to model's input format
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (256, 256))
            image = image / 127.5 - 1
            image = image.transpose(2, 0, 1)
            image = torch.tensor(image).unsqueeze(0)

            with torch.no_grad():
                output = self.model(image.float())

            # Convert output back to image
            output = output.squeeze(0).detach().numpy()
            output = output.transpose(1, 2, 0)
            output = (output + 1) * 127.5
            output = np.clip(output, 0, 255).astype(np.uint8)
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            # Encode the processed frame as base64
            output_image = Image.fromarray(output)
            buffer = BytesIO()
            output_image.save(buffer, format="JPEG")
            processed_frame = base64.b64encode(buffer.getvalue()).decode()

            # Send back the processed frame
            await websocket.send(processed_frame)

# Start the WebSocket server
async def start_server():
    processor = RealTimeJoJoStylization()
    async with websockets.serve(processor.process_frame, "localhost", 8000):
        await asyncio.Future()

if __name__ == '__main__':
    asyncio.run(start_server())
