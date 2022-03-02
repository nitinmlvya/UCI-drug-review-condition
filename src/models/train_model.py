from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from src.data.make_dataset import MakeDataSet
from src.features.build_features import BuildFeatures
from src.utils import save_obj_in_pickle
from src.visualization.visualizer import plot_confusion_matrix
import pandas as pd



class Trainer():
    """Class to train the model.
    """
    def __init__(self):
        """Constructore the trainer to load the training data and build the models.
        """
        self.data_file_path = 'data/raw_data/drugsComTrain_raw.tsv'
        self.model_file = 'data/processed_data/model.pkl'
        self.valid_cm_path = 'reports/figures/validation_confusion_matrix.png'
        
    
    def __over_sampling(self, X_train: object, y_train: object):
        """Do over sampling for the minority classes.

        Args:
            X_train (object): Features
            y_train (object): target data

        Returns:
            _type_: returns the oversampled training data and their labels.
        """

        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        return X_train, y_train
    
    
    def __split_data(self, X: object, y: object):
        """Split the data into training & validate

        Args:
            X (object): features
            y (object): target

        Returns:
            _type_: returns all the variants of the training and validation datasets.
        """
        # X_train, X_test, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=25, stratify=y)
        # X_train, y_train = self.__over_sampling(X_train, y_train)
        # return X_train, X_test, y_train, y_valid
        # X_train, X_test, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=25, stratify=y)
        X_train, y_train = self.__over_sampling(X, y)
        return X_train, None, y_train, None
    
    
    def train(self, X: object, y: object):
        """method to start the training and validation

        Args:
            X (object): features
            y (object): target
        """
        X_train, X_valid, y_train, y_valid = self.__split_data(X, y)
        self.clf = LinearSVC(random_state=25, class_weight='balanced', verbose=1)
        self.clf.fit(X_train, y_train)
        save_obj_in_pickle(self.clf, self.model_file) # save
        print(f'\n\nModel saved to file path: {self.model_file}')
        self.evaluate_model(X_train, X_valid, y_train, y_valid)
        
    
    def evaluate_model(self, X_train, X_valid, y_train, y_valid):
        y_train_pred = self.clf.predict(X_train)
        # y_valid_pred = self.clf.predict(X_valid)
        
        print('Training accuracy: ', accuracy_score(y_train, y_train_pred))
        # print('Test accuracy: ', accuracy_score(y_valid, y_valid_pred))
        
        print('\n\nClassification Report: ')
        # print(classification_report(y_valid, y_valid_pred))
        print(classification_report(y_train, y_train_pred))
        
        # cnf_matrix = confusion_matrix(le.inverse_transform(y_valid), le.inverse_transform(y_valid_pred), labels=le.classes_)
        print('\n\nConfusion Matrix: ')
        cnf_matrix = confusion_matrix(y_train, y_train_pred)
        plot_confusion_matrix(cnf_matrix, classes=list(set(y_train)), file_path=self.valid_cm_path)
        print(f'Confusion matrix saved to file path: {self.valid_cm_path}')
        

    def run(self):
        """method to run the multiclass classification.
        """
        print('Started model training...')

        print('--------------------------------')
        print('Data Cleansing...')
        source_df = pd.read_table(self.data_file_path)
        make_dataset = MakeDataSet(source_df)
        make_dataset.make_training_data()
        print('--------------------------------')
        print('\n\n')
        
        print('--------------------------------')
        print('Building features...')
        features = BuildFeatures(make_dataset.df_spacy_features)
        X = features.generate_features()
        y = features.df['condition']
        print('--------------------------------')
        print('\n\n')
        
        print('--------------------------------')
        print('Training model...')
        self.train(X, y)
        print('--------------------------------')
        print('\n\n')
        
        

if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()