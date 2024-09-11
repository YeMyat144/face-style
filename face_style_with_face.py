import os
import sys
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from argparse import Namespace
from facenet_pytorch import MTCNN

# Add JoJoGAN directory to sys.path
jojo_gan_path = r'D:/jupyter notebook/JoJoGAN'
if jojo_gan_path not in sys.path:
    sys.path.append(jojo_gan_path)

# Import necessary modules from the JoJoGAN project
try:
    from e4e.models.psp import pSp
    from e4e_projection import projection
    from model import Generator
except ModuleNotFoundError as e:
    print(f"Error: {e}. Ensure the 'e4e' module and other dependencies are present in {jojo_gan_path}")
    sys.exit()

# Initialize MTCNN for face detection with optimized parameters
mtcnn = MTCNN(keep_all=False, min_face_size=50, thresholds=[0.6, 0.7, 0.7], device='cuda' if torch.cuda.is_available() else 'cpu')

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Initialize webcam at a lower resolution for faster performance
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Unable to open webcam")
    sys.exit()

# Load the generator model
latent_dim = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

generator = Generator(1024, latent_dim, 8, 2).to(device)
style_name = "jojo"
preserve_color = False

ckpt = f'{style_name}.pt'
if preserve_color:
    ckpt = f'{style_name}_preserve_color.pt'

model_path = os.path.join(jojo_gan_path, 'models', ckpt)
try:
    ckpt = torch.load(model_path, map_location=device)
    generator.load_state_dict(ckpt["g"], strict=False)
    print(f"Loaded style model: {style_name}")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()

# Load the e4e model for projection
e4e_model_path = os.path.join(jojo_gan_path, 'models', 'e4e_ffhq_encode.pt')
if not os.path.exists(e4e_model_path):
    print(f"Error: The file {e4e_model_path} does not exist.")
    sys.exit()

try:
    ckpt = torch.load(e4e_model_path, map_location=device)
    opts = ckpt['opts']
    opts['checkpoint_path'] = e4e_model_path
    opts = Namespace(**opts)

    e4e_model = pSp(opts, device=device).to(device).eval()
    print("e4e model loaded successfully.")
except Exception as e:
    print(f"Error loading the e4e model: {e}")
    sys.exit()

generator.eval()

frame_count = 0  # To skip frames for faster detection

print("Press 'q' to quit the webcam")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from webcam. Exiting...")
        break

    # Display the webcam frame to check if the face is properly captured
    cv2.imshow("Webcam", frame)

    # Process every 5th frame to reduce CPU/GPU usage
    frame_count += 1
    if frame_count % 5 != 0:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Detect face using MTCNN
    aligned_face = mtcnn(frame)
    
    if aligned_face is None:
        print("Face not detected, skipping frame.")
        continue
    else:
        print(f"Face detected, shape: {aligned_face.shape}")

    # Convert the aligned face tensor to a NumPy array and then to a PIL Image
    aligned_face_np = aligned_face.permute(1, 2, 0).cpu().numpy()  # Convert Tensor to NumPy array
    aligned_face_pil = Image.fromarray((aligned_face_np * 255).astype(np.uint8))  # Convert NumPy to PIL Image

    # Resize and normalize the aligned face
    aligned_face_tensor = transform(aligned_face_pil).unsqueeze(0).to(device)
    print(f"Aligned face tensor shape: {aligned_face_tensor.shape}")

    with torch.no_grad():
        projected_latent = projection(aligned_face_pil, e4e_model, device).unsqueeze(0).to(device)

        # Generate stylized image
        stylized_image = generator(projected_latent, input_is_latent=True)
        stylized_image_np = stylized_image[0].cpu().permute(1, 2, 0).numpy()
        stylized_image_np = (stylized_image_np * 0.5 + 0.5) * 255
        stylized_image_np = stylized_image_np.astype(np.uint8)

        # Convert the image back to BGR for OpenCV display
        stylized_image_bgr = cv2.cvtColor(stylized_image_np, cv2.COLOR_RGB2BGR)
        cv2.imshow("Real-time Stylization", stylized_image_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
