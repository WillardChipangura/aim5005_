import numpy as np
from typing import List, Tuple

### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum=x.min(axis=0)
        self.maximum=x.max(axis=0)
        
    def transform(self, x:np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        
        # TODO: There is a bug here... Look carefully! 
        return (x-self.minimum)/(self.maximum-self.minimum)
    
    def fit_transform(self, x: List) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
    


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None  # Standard deviation
        
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
    
    def fit(self, x: List) -> None:
        x = self._check_is_array(x)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0, ddof=0)  # Use population standard deviation
        
    def transform(self, x: List) -> np.ndarray:
        x = self._check_is_array(x)
        return (x - self.mean) / self.std
    
    def fit_transform(self, x: List) -> np.ndarray:
        self.fit(x)
        return self.transform(x)
    
class LabelEncoder:
    def __init__(self):
        # After fitting, classes_ will be a sorted numpy array of unique labels.
        self.classes_ = None

    def fit(self, y) -> 'LabelEncoder':
        """
        Fit label encoder and store unique classes in sorted order.
        """
        y = np.array(y)
        self.classes_ = np.unique(y)
        return self

    def transform(self, y) -> np.ndarray:
        """
        Transform labels to normalized encoding.
        """
        y = np.array(y)
        # Build a mapping from label to integer index.
        mapping = {label: idx for idx, label in enumerate(self.classes_)}
        try:
            transformed = np.array([mapping[label] for label in y])
        except KeyError as e:
            raise ValueError(f"y contains unknown label: {e}")
        return transformed

    def fit_transform(self, y) -> np.ndarray:
        """
        Fit label encoder and return encoded labels.
        """
        return self.fit(y).transform(y)