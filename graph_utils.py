import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def get_subplot_dim(num:int)->Tuple[int,int]:
    """returns row and column dimensions closest
    to a square
    """
    r = int(np.floor(np.sqrt(num)))
    c = int(np.ceil(num/r))
    return r,c