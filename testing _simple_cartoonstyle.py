import cv2  
import numpy as np  
import tkinter as tk  
from threading import Thread  

# Function to apply cartoon effect  
def cartoonize_image(img):  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    gray = cv2.medianBlur(gray, 5)  
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)  
    color = cv2.bilateralFilter(img, 9, 250, 250)  
    cartoon = cv2.bitwise_and(color, color, mask=edges)  
    return cartoon  

# Function to toggle effect  
def toggle_effect():  
    global cartoon_mode  
    cartoon_mode = not cartoon_mode  
    button_text.set("Change to Normal" if cartoon_mode else "Change to Cartoon")  

# Function to capture video  
def capture_video():  
    global cartoon_mode  
    cap = cv2.VideoCapture(0)  

    while True:  
        ret, frame = cap.read()  
        if not ret:  
            break  
        
        if cartoon_mode:  
            frame = cartoonize_image(frame)  
        cv2.imshow('Video Feed', frame)  

        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  

    cap.release()  
    cv2.destroyAllWindows()  

cartoon_mode = False  

root = tk.Tk()  
root.title("Cartoon Effect Toggle")  

button_text = tk.StringVar()  
button_text.set("Change to Cartoon")  
toggle_button = tk.Button(root, textvariable=button_text, command=toggle_effect)  
toggle_button.pack(pady=20)  

video_thread = Thread(target=capture_video)  
video_thread.start()  

root.mainloop()