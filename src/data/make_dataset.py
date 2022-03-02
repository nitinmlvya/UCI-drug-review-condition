import numpy as np
import pandas as pd
import re
import html
import contractions
import spacy
from src.utils import read_pickle_obj, save_obj_in_pickle, save_to_csv_file, to_dataframe
from src.visualization.visualizer import conditions_plot


nlp = spacy.load('en_core_web_sm')


class MakeDataSet():
    """This is used to make the dataset from the raw data and do data cleansing.
    """
    def __init__(self, df, is_predict=False):
        """Constructor of the MakeDataSet

        Args:
            df (_type_): Pandas DataFrame having shape (n_samples, n_features)
            is_predict (bool, optional): Flag that is used to indicate training/prediction needs to be executed. Defaults to False.
        """
        self.is_predict = is_predict
        suffix = '_for_prediction' if self.is_predict else ''
        self.list_of_conditions_path ='data/processed_data/list_of_conditions_target.csv'
        self.spacy_features_path = f'data/processed_data/spacy_features{suffix}.pkl'
        self.spacy_features_csv_path = f'data/processed_data/spacy_features{suffix}.csv'
        self.df = df
        self.df_spacy_features = None


    def generate_top_conditions(self, plot_graph:bool=True):
        """Method to get top 10 conditions and mark remaining to "OTHERS" condition.

        Args:
            plot_graph (bool, optional): Allow to plot the graph for the conditions. Defaults to True.
        """
        top_10_conditions = self.df['condition'].value_counts().head(10).index.tolist()
        self.df['condition'] = self.df['condition'].apply(lambda x: x if x in top_10_conditions else 'OTHERS')
        #save
        save_obj_in_pickle(self.df['condition'].unique().tolist(), self.list_of_conditions_path)
        if plot_graph: conditions_plot(self.df, file_path='reports/figures/original_condition_counts.png')
    
    
    def remove_chars(self, text: str):
        """
            Remove all the unnecessary special characters.
        """
        text = html.unescape(text)
        text = contractions.fix(text)
        remove_chars_1 = r'[^A-Za-z0-9 ]'
        remove_chars_2 = r'^"'
        remove_chars_3 = r'"$'
        more_than_two_spaces = r'[\s\s]+'
        text = re.sub(remove_chars_1, ' ', text)
        text = re.sub(remove_chars_2, ' ', text)
        text = re.sub(remove_chars_3, ' ', text)
        text = re.sub(more_than_two_spaces, ' ', text)
        return text.strip()
    
    
    def features_generated_by_spacy_model(self, text: str) -> dict:
        """Generate features from the tokens of the sentences.

        Args:
            text (str): Sentence from the dataset.

        Returns:
            dict: returns the dictionary contains the detail information about the each token in the sentence.
        """
        text = nlp(text)
        token_features = {'norm_text': [], 'norm_text_counts': 0, 'noun_words': [], 'noun_counts': 0, 'stop_word_counts': 0, 'digits_counts': 0, 'lower_counts': 0, 'upper_counts': 0, 'title_counts': 0, 'total_words': 0, 'noun_phrase_counts': 0, 'noun_phrases': []}
        for token in text:
            if token.pos_ in ['NOUN', 'PROPN']:
                token_features['noun_counts'] += 1
                token_features['noun_words'].append(str(token))
            if token.is_digit:
                token_features['digits_counts'] += 1
            if token.is_lower and len(token) > 1:
                token_features['lower_counts'] += 1
            if token.is_upper and len(token) > 1:
                token_features['upper_counts'] += 1
            if token.is_title and len(token) > 1:
                token_features['title_counts'] += 1
            if token.is_stop:
                token_features['stop_word_counts'] += 1
            else:
                token_features['norm_text'].append(str(token.lemma_.lower()))

        token_features['total_words'] = len(text)
        token_features['noun_phrase_counts'] = len(list(text.noun_chunks))
        token_features['noun_phrases'] = [str(x) for x in list(text.noun_chunks)]
        token_features['norm_text_counts'] = len(token_features['norm_text'])
        token_features['norm_text'] = ' '.join(token_features['norm_text'])
        return token_features
    
    
    def spacy_features_to_df(self, spacy_features:dict) -> object:
        """Create pandas dataframe using the spacy features extracted for the sentences.

        Args:
            spacy_features (dict): token features.

        Returns:
            object: returns dataframe having the token information.
        """
        list_of_fs = []
        for x in spacy_features:
            f = {}
            f['norm_text'] = x['norm_text']
            f['norm_text_counts'] = x['norm_text_counts']
            f['noun_counts'] = x['noun_counts']
            f['stop_word_counts'] = x['stop_word_counts']
            f['digits_counts'] = x['digits_counts']
            f['total_words'] = x['total_words']
            f['noun_phrase_counts'] = x['noun_phrase_counts']
            list_of_fs.append(f)
        return to_dataframe(list_of_fs)
            
    
    
    def set_sentence_token_counts(self):
        """Method to set the length of the sentence token counts.
        """
        self.df_spacy_features['sentence_length'] = self.df_spacy_features["norm_text"].apply(lambda x: len(x.split()))
        
        
    def downsample_other_condition_records(self):
        """Remove some "others" condition samples that unnccessary
        """
        self.df_spacy_features = self.df_spacy_features[~((self.df_spacy_features['sentence_length'] <= 2) | (self.df_spacy_features['sentence_length'] > 100))].reset_index()

        other_downsampled_indexes1 = self.df_spacy_features[(self.df_spacy_features['condition'] == 'OTHERS') & ((self.df_spacy_features['sentence_length'] >= 30) & (self.df_spacy_features['sentence_length'] < 50 ))].sample(frac=1)[7000:].index.tolist()
        other_downsampled_indexes2 = self.df_spacy_features[(self.df_spacy_features['condition'] == 'OTHERS') & ((self.df_spacy_features['sentence_length'] >= 50) & (self.df_spacy_features['sentence_length'] < 70 ))].sample(frac=1)[7000:].index.tolist()
        other_downsampled_indexes3 = self.df_spacy_features[(self.df_spacy_features['condition'] == 'OTHERS')].sample(frac=1)[40000:].index.tolist()
        other_downsampled_indexes = list(set(other_downsampled_indexes1 + other_downsampled_indexes2 + other_downsampled_indexes3))
        self.df_spacy_features.drop(self.df_spacy_features.index[other_downsampled_indexes], inplace=True) # Drop indexes


    def make_training_data(self):
        """Method to make the training data available.
        """
        self.generate_top_conditions()
        self.df['cleaned_text'] = self.df['review'].apply(self.remove_chars)
        
        spacy_features = read_pickle_obj(self.spacy_features_path)
        if spacy_features is False:
            spacy_features = self.df['cleaned_text'].apply(self.features_generated_by_spacy_model)
            spacy_features = [x for x in spacy_features]
            save_obj_in_pickle(spacy_features, self.spacy_features_path)

        self.df_spacy_features = self.spacy_features_to_df(spacy_features)
        self.df_spacy_features['condition'] = self.df['condition']
        
        save_to_csv_file(self.df_spacy_features, file_path=self.spacy_features_csv_path)
        print(f'Saved spacy features to {self.spacy_features_csv_path}')
        self.set_sentence_token_counts()
        self.downsample_other_condition_records()
        print('Downsampled done...')
        
        # save
        conditions_plot(self.df_spacy_features, file_path='reports/figures/condition_counts_after_downsampling_others.png')


    def make_data_for_prediction(self):
        """Method to allow data generation for the dataset.
        """
        conditions_list = read_pickle_obj(self.list_of_conditions_path)
        print('List of conditions: ', conditions_list)
            
        self.df['cleaned_text'] = self.df['review'].apply(self.remove_chars)
        
        print('features path: ', self.spacy_features_path)
        spacy_features = read_pickle_obj(self.spacy_features_path)
        if spacy_features is False:
            spacy_features = self.df['cleaned_text'].apply(self.features_generated_by_spacy_model)
            spacy_features = [x for x in spacy_features]
            save_obj_in_pickle(spacy_features, self.spacy_features_path)

        self.df_spacy_features = self.spacy_features_to_df(spacy_features)
        self.df_spacy_features['condition'] = self.df['condition'].apply(lambda x: x if x in conditions_list else 'OTHERS')
        
        save_to_csv_file(self.df_spacy_features, file_path=self.spacy_features_csv_path)
        print(f'Saved spacy features to {self.spacy_features_csv_path}')
        # save
        conditions_plot(self.df_spacy_features, file_path='reports/figures/condition_counts_after_downsampling_others_for_prediction.png')
