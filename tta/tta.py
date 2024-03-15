import abc
import numpy as np


class TTATransformation(abc.ABC):
    def forward_transform(self, arr):
        pass
    
    def backward_transform(self, arr):
        pass
    
    
class TTACompose(TTATransformation):
    def __init__(self, transformations):
        self.transformations = transformations
        
    def forward_transform(self, arr):
        result = arr.copy()
        for transform in self.transformations:
            result = transform.forward_transform(result)
        return result
    
    def backward_transform(self, arr):
        result = arr.copy()
        for transform in self.transformations[::-1]:
            result = transform.backward_transform(result)
        return result
    
    
class TTAIdentity(TTATransformation):
    def forward_transform(self, arr):
        return arr.copy()
    
    def backward_transform(self, arr):
        return arr.copy()
    
    
class TTARotation(TTATransformation):
    def __init__(self, k):
        self.k = k
        
    def forward_transform(self, arr):
        result = arr.copy()
        result = np.rot90(result, k=self.k, axes=(0, 1))
        return result
    
    def backward_transform(self, arr):
        result = arr.copy()
        result = np.rot90(result, k=-self.k, axes=(0, 1))
        return result
    
        
class TTARot90(TTARotation):
    def __init__(self):
        super(TTARot90, self).__init__(k=1)
    
    
class TTARot180(TTARotation):
    def __init__(self):
        super(TTARot180, self).__init__(k=2)
    
    
class TTARot270(TTARotation):
    def __init__(self):
        super(TTARot270, self).__init__(k=3)
    
    
class TTAFlipHorizontally(TTATransformation):
    def forward_transform(self, arr):
        result = arr.copy()
        return result[:, ::-1, :]
    
    def backward_transform(self, arr):
        return self.forward_transform(arr)
    
    
class TTAFlipVertically(TTATransformation):
    def forward_transform(self, arr):
        result = arr.copy()
        return result[::-1, :, :]
    
    def backward_transform(self, arr):
        return self.forward_transform(arr)
