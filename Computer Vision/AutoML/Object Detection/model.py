!pip install --pre autogluon 
!pip install autogluon
%matplotlib inline
import autogluon.core as ag
from autogluon.vision import ObjectDetector

warnings.filterwarnings('ignore')
def modeling(train_path,logging=False):
  #Tiny_motorbike Dataset
  data = train_path
  dataset_train = ObjectDetector.Dataset.from_voc(data, splits='trainval')
  dataset_train
  
  #Fit Models by AutoGluon
  time_limit = 60*30  # at most 0.5 hour
  detector = ObjectDetector()
  hyperparameters = {'epochs': 5, 'batch_size': 8}
  hyperparamter_tune_kwargs={'num_trials': 2}
  detector.fit(dataset_train, time_limit=time_limit, hyperparameters=hyperparameters, hyperparamter_tune_kwargs=hyperparamter_tune_kwargs)
  
  #Test
  dataset_test = ObjectDetector.Dataset.from_voc(data, splits='test')
  test_map = detector.evaluate(dataset_test)
  print("mAP on test dataset: {}".format(test_map[1][-1]))
  
  #Below, we randomly select an image from test dataset and show the predicted class, box and probability over the origin image, stored in predict_class, predict_rois and predict_score columns, respectively.
  #You can interpret predict_rois as a dict of (xmin, ymin, xmax, ymax) proportional to original image size.
  image_path = dataset_test.iloc[0]['image']
  result = detector.predict(image_path)
  print(result)
  
  #Prediction with multiple images is permitted:
  bulk_result = detector.predict(dataset_test)
  print(bulk_result)




