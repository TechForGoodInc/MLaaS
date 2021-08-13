
# Image Classification using AutoGluon
![image](https://user-images.githubusercontent.com/53899191/127358834-4bbab40a-f0c5-4760-96b2-14e65f13020a.png)


This is different from traditional machine learning where we need to manually define the neural network and then specify the hyperparameters in the training process. 
Instead, with just a single call to AutoGluon’s fit function, AutoGluon automatically trains many models with different hyperparameter configurations and returns the model that achieved the highest level of accuracy.

## Dataset

You can use the sample Image Dataset, below:
- [Grocery Data](https://www.kaggle.com/khotijahs1/grocery-data)
- [Monkey Species Data](https://www.kaggle.com/khotijahs1/monkey-species-dataset)
- [Shopeeiet Dataset](https://www.kaggle.com/khotijahs1/shopeeiet-dataset)

## Sample Image
![image](https://user-images.githubusercontent.com/53899191/127365383-03500e3f-e455-49bd-a73d-fea130045902.png)


## AutoGluon
```
import autogluon.core as ag
from autogluon.vision import ImagePredictor, ImageDataset

#Create AutoGluon Dataset
train_dataset, _, test_dataset = ImageDataset.from_folders('../data/grocery20/')
print(train_dataset)

#Use AutoGluon to Fit Models
predictor = ImagePredictor()
predictor.fit(train_dataset, hyperparameters={'epochs': 2})  # you can trust the default config, we reduce the # epoch to save some build time

#result
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
```

## Sample Output

```
#print(proba)
    0         1         2         3
0  0.555801  0.413695  0.022135  0.008369

#print(bulk_result)

0     0
1     0
2     0
3     2
4     1
     ..
75    3
76    3
77    3
78    3
79    3
Name: label, Length: 80, dtype: int64
```

## Reference :

This section provides more resources on the topic if you are looking to go deeper.

- https://github.com/awslabs/autogluon
