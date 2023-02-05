
import mlflow
import mlflow.tensorflow

from preprocessing import load_dataset, prep_pixels
from model_creation import define_model


# run the test harness for evaluating a model
def main(run_name, tracking_id, experiment_name):

 with  mlflow.start_run(run_name=run_name):
   mlflow.set_tracking_uri(tracking_id)

   mlflow.tensorflow.autolog()

   # load dataset
   trainX, trainY, testX, testY = load_dataset()

   # prepare pixel data
   trainX, testX = prep_pixels(trainX, testX)

   # define model
   model = define_model()

   # fit model
   model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)

   # save model
   model.save('final_model.h5')
   run_id = mlflow.active_run().info.run_id
   mlflow.end_run()



if __name__ == "__main__":

 run_name='iter1'
 tracking_id = "http://localhost:5000"
 experiment_name = "[M] Mnist - Training-model - Ankush "

 main(run_name, tracking_id, experiment_name)

