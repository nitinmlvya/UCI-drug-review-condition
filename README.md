## Python version 3.7.

### Install necessary packages.
```
pip install -r requirements.txt

python -m spacy download en
```




## Dataset

[UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/ml/datasets/drug+review+dataset+drugs+com)

### Task

Given a review, predict the underlying condition.

### Metrics

1. Accuracy
2. Confusion matrix (for top 10 classes in the dataset only, consider all other classes as ‘other’)


<br><br>


### Download the dataset and unzip it. 
```
!wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
!unzip "drugsCom_raw.zip"
```

#### Place the dataset in the directory (data/raw_data)


### Train the model
```
python -m src.models.train_model
```

### Predict and evaluate the test dataset
```
python -m src.models.predict_model
```

### Model evaluation metrics for test data 
```
reports/results.docx
```
