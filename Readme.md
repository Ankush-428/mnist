<h1>Digit Image Prediction </h1>

<h2>Steps to run (Without Docker)</h2>
<ul>
<li>Command: Python api.py</li>
<li>Trigger endpoint through PostMan </li>
<ul><li> Endpoint: 127.0.0.1:9090/predict</li></ul>

Steps: 
1. Train.py used to train the model with mlflow  
2. Predict.py file is used for prediction after loading model
3. Microservice folder contains api.py, docker,deployment.yaml where api.py file use to hit the apai where we request the image from postman and predict the image using the model and identify the digit.
4. Containerize whole code in docker and build the image using ~docker build -t microservice .
5. After running the container hit the api with mapping to new address.
6. Login to docker hub and tag the image and push to docker hub.
7. Using deployment.yaml file we can deploy the image into k8 with kubectl command.
    
</ul>