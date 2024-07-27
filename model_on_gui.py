import model
import tkinter as tk
from tkinter import Button, messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Load the trained TensorFlow model
model = model.init()

class DigitClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Classifier")

        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()

        self.button_classify = Button(root, text="Classify", command=self.classify_digit)
        self.button_classify.pack()

        self.button_clear = Button(root, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        self.image = Image.new('L', (280, 280), color='white')
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
    
    def paint(self, event):
        x, y = event.x, event.y
        self.draw.ellipse([x-15, y-15, x+15, y+15], fill='black')
        self.canvas.create_oval(x-15, y-15, x+15, y+15, fill='black')

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        # Clear the canvas and reset the image
        self.canvas.delete("all")
        self.image = Image.new('L', (280, 280), color='white')
        self.draw = ImageDraw.Draw(self.image)

    def classify_digit(self):
        # Convert the drawn image to 28x28 pixels, inverted color, and normalized
        img = ImageOps.invert(self.image).resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape((1, 28, 28, 1))

        # Predict the digit
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        # Display the result
        messagebox.showinfo("Prediction", f"Predicted digit: {predicted_digit}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitClassifierApp(root)
    root.mainloop()
