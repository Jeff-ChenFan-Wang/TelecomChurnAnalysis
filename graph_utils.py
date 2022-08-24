import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def get_subplot_dim(num:int)->Tuple[int,int]:
    r = int(np.ceil(np.sqrt(num)))
    c = int(np.ceil(num/r))
    return r,c
    # for i in range(c):
    #     for j in range(c):
    #         if i*c+j<num:
    #             cluster = classes[i*c+j]
    #             sns_graph(
    #                 x=kwargs.get('x',None),
    #                 y=kwargs.get('y',None),
    #                 ax=ax[i][j]
    #             )
    #             ax[i][j].title.set_text(f'cluster {cluster}')