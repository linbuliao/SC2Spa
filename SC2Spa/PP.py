import numpy as np
import pandas as pd

pct_offset = 0

def MinMaxNorm(Y:np.array, Y_ref:np.array):
    '''
    Min-Max normalize along the second axis

    Parameters
    ----------
    Y
        The coordinates to be normalized
    Y_ref
        The reference coordinates used to determine the minimum and maximum values

    Returns
    -------
    Min-Max Normalized Y

    '''
    return (Y-Y_ref.min(axis = 0))/(Y_ref.max(axis = 0)-Y_ref.min(axis = 0))/(1+pct_offset)

def ReMMNorm(Y_ref:np.array, Y_pred:np.array):
    '''
    Reverse Min-Max normalize along the second axis

    Parameters
    ----------
    Y_ref
        The reference coordinates used to determine the minimum and maximum values
    Y_pred
        The coordinates to be reversely Min-Max normalized

    Returns
    -------
    Reversely Min-Max Normalized Y

    '''

    return (Y_pred*(Y_ref.max(axis=0)-Y_ref.min(axis=0)*(1+pct_offset))+Y_ref.min(axis = 0))

def PolarTrans(Y):
    '''
    Transform cartesian coordinates to polar coordinates

    Parameters
    ----------
    Y
        The cartesian coordinates to be transformed

    Returns
    -------
    RTheta
        Polar coordinates of Y

    '''
    
    R = np.sqrt(np.square(Y[:,0]) + np.square(Y[:,1]))
    Theta = np.arctan(Y[:,1]/Y[:,0])
    RTheta = np.concatenate([R.reshape(-1,1), Theta.reshape(-1,1)], axis = 1)
    
    return RTheta

def RePolarTrans(RTheta):
    '''
    Transform polar coordinates to cartesian coordinates

    Parameters
    ----------
    RTheta
        The polar coordinates to be transformed

    Returns
    -------
    Y
        Cartesian coordinates of RTheta

    '''

    x = RTheta[:,0] * np.cos(RTheta[:,1])
    y = RTheta[:,0] * np.sin(RTheta[:,1])
    Y = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis = 1)
    return Y
    
def LoadIndices(prefix = ''):
    '''
    Load training and test indices for cross-validation from
    prefix + 'CV_groups/index_train_' + str(i+1) + '.csv', i = 0,1,2,3,4
    prefix + 'CV_groups/index_test_' + str(i+1) + '.csv', i = 0,1,2,3,4

    Parameters
    ----------
    prefix

    Returns
    -------
    train_indices
        a list contains mutiple lists of the index of cells for training
    test_indices
        a list contains mutiple lists of the index of cells for test. The train_indices
         and test_indices are paired based on the order of indices list.
    '''

    train_indices = []
    test_indices = []

    for i in range(5):
        train_index = pd.read_csv(prefix + 'CV_groups/index_train_' + str(i+1) + '.csv',
                                  header = None, index_col = 0).values.flatten()
        test_index = pd.read_csv(prefix + 'CV_groups/index_test_' + str(i+1) + '.csv',
                                 header = None, index_col = 0).values.flatten()
        train_indices.append(train_index)
        test_indices.append(test_index)
        
    return train_indices, test_indices
