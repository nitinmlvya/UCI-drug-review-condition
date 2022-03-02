from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.data.make_dataset import MakeDataSet
from src.features.build_features import TransformFeatures
import pandas as pd

from src.utils import read_pickle_obj
from src.visualization.visualizer import plot_confusion_matrix

class PredictModel():
    """Class of the PredictModel
    """
    def __init__(self,):
        """Constructor of the PredictModel
        """
        self.data_file_path = 'data/raw_data/drugsComTest_raw.tsv'
        self.model_file = 'data/processed_data/model.pkl'
        self.vectorizer_path = 'data/processed_data/tfidf_vectorizer.pkl'
        self.valid_cm_path = 'reports/figures/validation_confusion_matrix_for_prediction.png'
        

    def load_model(self):
        """Load the model and store into variable.
        """
        self.clf = read_pickle_obj(self.model_file)
        
    
    def evaluate_model(self, X: object, y: object):
        """Evalute the model accuracy and validate the results.

        Args:
            X (object): Features
            y (object): target
        """
        y_pred = self.clf.predict(X)
        
        print('Test accuracy: ', accuracy_score(y, y_pred))
        
        print('\n\nClassification Report: ')
        print(classification_report(y, y_pred))
        
        # cnf_matrix = confusion_matrix(le.inverse_transform(y_valid), le.inverse_transform(y_valid_pred), labels=le.classes_)
        print('\n\nConfusion Matrix: ')
        cnf_matrix = confusion_matrix(y, y_pred)
        plot_confusion_matrix(cnf_matrix, classes=list(set(y)), file_path=self.valid_cm_path)
        print(f'Confusion matrix saved to file path: {self.valid_cm_path}')


    def run(self):
        """Method to execute the prediction on the data set.
        """
        print('Started model training...')

        print('--------------------------------')
        print('Data Cleansing...')
        source_df = pd.read_table(self.data_file_path)
        make_dataset = MakeDataSet(source_df, is_predict=True)
        make_dataset.make_data_for_prediction()
        print('--------------------------------')
        print('\n\n')

        print('--------------------------------')
        print('Building features & load models...')
        features = TransformFeatures(make_dataset.df_spacy_features)
        X = features.transform_features()
        y = features.df['condition']
        print('--------------------------------')
        print('\n\n')

        print('--------------------------------')
        print('Predict & Evaluate results...')
        self.load_model()
        self.evaluate_model(X, y)
        print('--------------------------------')
        print('\n\n')


if __name__ == '__main__':
    predict_model = PredictModel()
    predict_model.run()