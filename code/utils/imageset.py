# Module for handling MIT_CBCL dataset
import pickle
from torch.utils.data import Dataset

def save_dataset(dataset, file):
"""
Save a configuration of an augmented dataset to a location to speed up development

Args
    dataset: a dictionary with properties 'data' and 'labels'
    file: filepath of where to save pickled data
"""    
    pickle.dump(dataset, open(location, "wb"))


    
def load_dataset(file):
"""
Load a dataset from file location

Args
    file: filepath to  pickled binary

Returns
    A dict of unpickled data
"""    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



class MIT-CBCL(Dataset):
"""
Overloaded class for handling MIT-CBCL dataset with pytorch methods

Args
    Dataset: filepath of pickled dataset file
"""

    def __init__(self, file):
        batch = load_dataset(file)
        self.data = batch[b'data']
        self.labels = batch[b'labels']

    def  __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        # image normalization here?
        target = self.labels[idx]
        return (image, target)
