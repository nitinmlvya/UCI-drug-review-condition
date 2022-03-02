import pickle
from pathlib import Path
import pandas as pd


def to_dataframe(data: list) -> object:
    """Convert the list of dictionary into dataframe

    Args:
        data (list): list of the sentences

    Returns:
        object: returns the dataframe.
    """
    return pd.DataFrame(data)


def save_to_csv_file(df: object, file_path:str):
    """Save the CSV file

    Args:
        df (object): Pandas dataframe having huge information
        file_path (str): Path to the file path.
    """
    df.to_csv(file_path, index=False)
    

def save_obj_in_pickle(data: object, file_path:str):
    """method to save the pickle

    Args:
        data (object): DataFrame
        file_path (str): path of the file to save
    """
    file_to_store = open(file_path, "wb")
    pickle.dump(data, file_to_store)
    file_to_store.close()
    
    
def read_pickle_obj(file_path: str):
    """Returns the list of dictionary havin 

    Args:
        file_path (str): read the pickle file

    Returns:
        _type_: _description_
    """
    file_path = Path(file_path)
    if file_path.is_file():
        with open(file_path, "rb") as fp:
            data = pickle.load(fp)
            fp.close()
            return data
    else:
        return False