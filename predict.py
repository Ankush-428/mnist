import mlflow
import mlflow.tensorflow
from numpy import argmax
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model

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

# load an image and predict the class
def predict_image():

 with mlflow.start_run(run_name='iter2'):
  mlflow.set_tracking_uri("http://localhost:5000")
  mlflow.set_experiment("[M] Mnist - prediction-model - Ankush ")

  img = load_image('sample_image.png')
  # load model
  model = load_model('final_model.h5')
  # predict the class
  predict_value = model.predict(img)
  digit = argmax(predict_value)
  mlflow.log_param("digit", digit)
  mlflow.tensorflow.log_model(model, 'model')
  mlflow.end_run()


if __name__ == "__main__":
 predict_image()

