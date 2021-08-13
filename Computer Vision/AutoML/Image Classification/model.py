!pip install --pre autogluon 
!pip install autogluon
import autogluon.core as ag
from autogluon.vision import ImagePredictor, ImageDataset

warnings.filterwarnings('ignore')
def modeling(train_path,test_path,logging=False):

  #Create AutoGluon Dataset
  train_dataset, _, test_dataset = ImageDataset.from_folders(train_path,test_path)
  print(train_dataset)
  
  #Use AutoGluon to Fit Models
  predictor = ImagePredictor()
  # since the original dataset does not provide validation split, the `fit` function splits it randomly with 90/10 ratio
  predictor.fit(train_dataset, hyperparameters={'epochs': 50})  # you can trust the default config, we reduce the # epoch to save some build time
  fit_result = predictor.fit_summary()
  print('Top-1 train acc: %.3f, val acc: %.3f' %(fit_result['train_acc'], fit_result['valid_acc']))
  
  #Predict on a New Image
  image_path = test_dataset.iloc[0]['image']
  result = predictor.predict(image_path)
  print(result)
  
  #If probabilities of all categories are needed, you can call predict_proba:
  proba = predictor.predict_proba(image_path)
  print(proba)
  
  #You can also feed in multiple images all together, let’s use images in test dataset as an example:
  bulk_result = predictor.predict(test_dataset)
  print(bulk_result)
  
  #Generate image features with a classifier
  image_path = test_dataset.iloc[0]['image']
  feature = predictor.predict_feature(image_path)
  print(feature)
  
  #Evaluate on Test Dataset
  test_acc, _ = predictor.evaluate(test_dataset)
  print('Top-1 test acc: %.3f' % test_acc)
  
  #Save and load classifiers
  filename = 'predictor.ag'
  predictor.save(filename)
  predictor_loaded = ImagePredictor.load(filename)
  # use predictor_loaded as usual
  result = predictor_loaded.predict(image_path)
  print(result)




  

