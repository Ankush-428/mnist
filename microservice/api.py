from PIL.Image import Image
from numpy import argmax
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, request
import tensorflow as tf


app = Flask(__name__)

model = load_model('final_model.h5')

# load and prepare the image
def load_image(filename):
 # load the image
 img = load_img(filename, grayscale=True, target_size=(28, 28))
 # convert to array
 img = img_to_array(img)
 # reshape into a single sample with 1 channel
 img = img.reshape(1, 28, 28, 1)
 # prepare pixel data
 img = img.astype('float32')
 img = img / 255.0
 return img

@app.route("/predict", methods=["POST"])
def predict():

    image = request.files.get("image")

    # Check if the image was received
    if image is None:
        return "No image received", 400
    image.save("test.png")
    img = load_image("../test.png")
    predictions = model.predict(img)
    digit = argmax(predictions)
    return {"predictions": str(digit)}

if __name__ == '__main__':
    print(app.run(debug=True, host='0.0.0.0', port=9090))

