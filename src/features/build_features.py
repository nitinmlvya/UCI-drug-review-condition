from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from src.utils import read_pickle_obj, save_obj_in_pickle


class BuildFeatures():
    """Class to build features
    """
    def __init__(self, df: object):
        """_summary_

        Args:
            df (object): Constructor of the BuildFeatures.
        """
        self.df = df
        self.vectorizer_path = 'data/processed_data/tfidf_vectorizer.pkl'
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.5)
        # self.vectorizer = HashingVectorizer(ngram_range=(1, 3), n_features=2 ** 20)


    def generate_features(self) -> object:
        """Generate features for the text.

        Returns:
            object: features data structure.
        """
        transformed_data = self.vectorizer.fit_transform(self.df['norm_text'])
        save_obj_in_pickle(self.vectorizer, self.vectorizer_path)
        return transformed_data
    
    
class TransformFeatures():
    """Class to transform the data into features.
    """
    def __init__(self, df: object):
        """Constructor

        Args:
            df (object): Pandas Dataframe.
        """
        self.df = df
        self.vectorizer_path = 'data/processed_data/tfidf_vectorizer.pkl'
        self.vectorizer = read_pickle_obj(self.vectorizer_path)
        print(self.vectorizer)
        
    
    def transform_features(self) -> object:
        """method to transform the samples into features

        Returns:
            object: returns the transformed features for the given text.
        """
        return self.vectorizer.transform(self.df['norm_text'])