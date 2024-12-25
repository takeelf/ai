from sklearn.datasets import load_iris
from torch.utils.data import Dataset
import pandas as pd

class IrisDataset(Dataset):
    def __init__(self, train=True):
        dataset = load_iris()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset.data, dataset.target, test_size=0.3, random_state=827
        )        
