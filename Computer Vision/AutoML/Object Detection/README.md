
# Object Detection using AutoGluon

![image](https://user-images.githubusercontent.com/53899191/127372565-85fec402-5ba8-4dfb-a5fd-873490b31053.png)

Object detection is the process of identifying and localizing objects in an image and is an important task in computer vision.

Our goal is to detect motorbike in images by [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) model. A tiny dataset is collected from VOC dataset, which only contains the motorbike category. The model pretrained on the COCO dataset is used to fine-tune our small dataset. With the help of AutoGluon, we are able to try many models with different hyperparameters automatically, and return the best one as our final model.

## Dataset

You can use the sample Dataset, below:
- [Tiny Dataset](https://www.kaggle.com/khotijahs1/tiny-dataset)

## Sample Image
![image](https://user-images.githubusercontent.com/53899191/127374415-b12beb87-d47b-4c7e-bdd0-649ff373562b.png)


## AutoGluon
```
# !pip install --pre autogluon 
!pip install autogluon
%matplotlib inline
import autogluon.core as ag
from autogluon.vision import ObjectDetector

#Tiny_motorbike Dataset
data = '../data/tiny_motorbike'
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

# result
image_path = dataset_test.iloc[0]['image']
result = detector.predict(image_path)
print(result)

#Prediction with multiple images is permitted:
bulk_result = detector.predict(dataset_test)
print(bulk_result)
```

## Sample Output

### result
```
predict_class  predict_score  \
0      motorbike       0.988112   
1         person       0.970208   
2        bicycle       0.294567   
3         person       0.225033   
4         person       0.224180   
..           ...            ...   
95        person       0.032129   
96        person       0.032065   
97        person       0.032060   
98        person       0.031963   
99        person       0.031809   

                                         predict_rois  
0   {'xmin': 0.32461073994636536, 'ymin': 0.445641...  
1   {'xmin': 0.4067949950695038, 'ymin': 0.2784966...  
2   {'xmin': 0.0, 'ymin': 0.6488332152366638, 'xma...  
3   {'xmin': 0.3642578125, 'ymin': 0.2946056425571...  
4   {'xmin': 0.33951500058174133, 'ymin': 0.437841...  
..                                                ...  
95  {'xmin': 0.3480450212955475, 'ymin': 0.1806999...  
96  {'xmin': 0.7947438359260559, 'ymin': 0.3857879...  
97  {'xmin': 0.8372661471366882, 'ymin': 0.3870421...  
98  {'xmin': 0.8781052827835083, 'ymin': 0.0032495...  
99  {'xmin': 0.8558664321899414, 'ymin': 0.3969009...  

[100 rows x 3 columns]
```
### print(bulk_result)
```
predict_class  predict_score  \
0        motorbike       0.988112   
1           person       0.970208   
2          bicycle       0.294567   
3           person       0.225033   
4           person       0.224180   
...            ...            ...   
4594        person       0.034571   
4595        person       0.034512   
4596        person       0.034480   
4597        person       0.034463   
4598        person       0.034428   

                                           predict_rois  \
0     {'xmin': 0.32461073994636536, 'ymin': 0.445641...   
1     {'xmin': 0.4067949950695038, 'ymin': 0.2784966...   
2     {'xmin': 0.0, 'ymin': 0.6488332152366638, 'xma...   
3     {'xmin': 0.3642578125, 'ymin': 0.2946056425571...   
4     {'xmin': 0.33951500058174133, 'ymin': 0.437841...   
...                                                 ...   
4594  {'xmin': 0.015702269971370697, 'ymin': 0.35812...   
4595  {'xmin': 0.09243986010551453, 'ymin': 0.338881...   
4596  {'xmin': 0.014295696280896664, 'ymin': 0.36919...   
4597  {'xmin': 0.354862242937088, 'ymin': 0.39860683...   
4598  {'xmin': 0.929229736328125, 'ymin': 0.11419545...   

                                                  image  
0     ../input/tiny-dataset/tiny_motorbike/JPEGImage...  
1     ../input/tiny-dataset/tiny_motorbike/JPEGImage...  
2     ../input/tiny-dataset/tiny_motorbike/JPEGImage...  
3     ../input/tiny-dataset/tiny_motorbike/JPEGImage...  
4     ../input/tiny-dataset/tiny_motorbike/JPEGImage...  
...                                                 ...  
4594  ../input/tiny-dataset/tiny_motorbike/JPEGImage...  
4595  ../input/tiny-dataset/tiny_motorbike/JPEGImage...  
4596  ../input/tiny-dataset/tiny_motorbike/JPEGImage...  
4597  ../input/tiny-dataset/tiny_motorbike/JPEGImage...  
4598  ../input/tiny-dataset/tiny_motorbike/JPEGImage...  

[4599 rows x 4 columns]
```

## Sample Output Image
![image](https://user-images.githubusercontent.com/53899191/127374377-31926a5a-ad98-45eb-a529-2078a9d297d8.png)


## Reference :

This section provides more resources on the topic if you are looking to go deeper.

- https://github.com/awslabs/autogluon
