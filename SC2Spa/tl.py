import numpy as np
import pandas as pd
import os
import gc
import pickle
import anndata

from sklearn.neighbors import NearestNeighbors
from scipy.stats import wasserstein_distance

import tensorflow as tf
from keras.models import Model, load_model
from keras.regularizers import l1_l2
from keras.layers import Dense, Input, BatchNormalization, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from IPython.display import clear_output

import keras.backend as K
from . import pp

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
def calc_rmse(y_true, y_pred):
    return np.sqrt(np.square(y_true - y_pred).mean())

loss = rmse
loss_name = 'rmse'

def FCNN(input_shape: tuple, out_shape: int,l1_reg = 1e-7, l2_reg = 0,
	     dropout = 0, nodes = [4096, 1024, 256, 64, 16, 4], seed = None):
    '''
    Construct a fully connected neural network

    Parameters
    ---------
    input_shape
        a tuple with the shape of (number of genes, )
    out_shape
        the dimension of coordinates
    dropout
        dropout rate for all hidden layers and the input layer
    nodes
        a list that contains the numbers of the nodes of hidden layers

    Returns
    -------
    model
        a fully connected neural network model

    '''
    
    X_input = Input(input_shape)    
    X = X_input
    
    for node in nodes:
        X = Dropout(rate = dropout, seed = seed)(X)
        X = Dense(node, kernel_initializer = tf.keras.initializers.GlorotNormal(seed = seed),
                  activation = 'relu', kernel_regularizer = l1_l2(l1_reg, l2_reg))(X)
        X = BatchNormalization(axis = 1)(X)
    
    X = Dense(out_shape, kernel_initializer = tf.keras.initializers.GlorotNormal(seed = seed),
              activation = 'sigmoid')(X)
    
    model = Model(inputs = X_input, outputs = X, name = 'FCNN')
    
    return model
    
def batch_generator(X:np.array, y: np.array, sample_index:list, batch_size: int, shuffle: bool):
    '''
    Split data for batch training

    Parameters
    ----------
    X
        a numpy array with the shape of (cell, gene)
    Y
        a numpy array with the shape of (cell, dimension), where dimension is
         the dimension of spatial information
    sample_index
        indices of cells for training
    batch_size
        number of cells for one batch
    shuffle
        if shuffle the indices before generating batch data

    Yields
    -------
    X_batch
        gene expression array for one batch
    y_batch
        coordinate array for one batch

    '''
    number_of_batches = int(np.ceil(len(sample_index)/batch_size))
    counter = 0

    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index, :]
        y_batch = y[batch_index, :]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def CrossValidation(X: np.array, Y: np.array, train_indices, test_indices,
                    l1_reg = 1e-5, l2_reg = 0, dropout = 0.05, lrr_patience = 20,
                    ES_patience = 50, min_lr = 1e-9,
                    epoch = 20, batch_size = 4096,
                    nodes = [4096, 1024, 256, 64, 16, 4], seed = None):
    '''
    Perform Cross-validation using fully connected neural network

    Parameters
    ----------
    X
        a numpy array with the shape of (cell, gene)
    Y
        a numpy array with the shape of (cell, dimension), where dimension is
         the dimension of spatial information
    train_indices
        a list contains mutiple lists of the index of cells for training
    test_indices
        a list contains mutiple lists of the index of cells for test. The train_indices
         and test_indices are paired based on the order of indices list.
    l1_reg
        l1 regularization factor
    nodes
        a list that contains the numbers of the nodes of hidden layers
    lrr_patience
        The patience for learning rate reduction
    ES_patience
        The patience for early stopping

    Returns
    -------
    histories
        training history
    train_preds
        predicted locations of cells for training, which corresponds to train_indices.
    test_preds
        predicted locations of cells for test, which corresponds to test_indices

    '''

    length = Y.shape[1]

    earlystopper = EarlyStopping(patience=ES_patience, verbose=1)

    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_' + loss_name, 
                                                patience=lrr_patience,
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=min_lr)
    
    histories = []
    train_preds = []
    test_preds = []

    for i in range(5):
        
        print(i)
        cv_train = train_indices[i].copy()
        cv_test = test_indices[i].copy()

        X_test = X[cv_test, :].copy()
        X_train = X[cv_train, :]

        model = FCNN((X.shape[1],), length, nodes = nodes, dropout = dropout,
                     l1_reg =  l1_reg, l2_reg = l2_reg, seed = seed)
        model.compile(optimizer = 'adam', loss = loss, metrics = [loss])

        if(seed == None):
            shuffle = True
        else:
            shuffle = False
        history = model.fit_generator(generator = batch_generator(X, Y, cv_train,
                                                                batch_size, shuffle),
                                    epochs = epoch, steps_per_epoch = len(cv_train) / batch_size,
                                    validation_data = (X_test, Y[cv_test, :]),
                                    callbacks = [learning_rate_reduction, earlystopper])
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_preds.append(train_pred)
        test_preds.append(test_pred)
        histories.append(history)
        gc.collect()
    
    return histories, train_preds, test_preds

def Train(X, Y, root='Model_SI/', name = 'SI', 
	      l1_reg = 1e-5, l2_reg = 0, dropout = 0.05,
          epoch = 500, batch_size = 4096,
          nodes = [4096, 1024, 256, 64, 16, 4], lrr_patience = 20,
          ES_patience = 50, min_lr = 1e-5, save = True, seed = None):
    '''
    Train a fully connected neural network.
    The model will be saved to `root+name+'.h5'`

    Parameters
    ----------
    X
        a numpy array with the shape of (cell, gene)
    Y
        a numpy array with the shape of (cell, dimension), where dimension is
         the dimension of spatial information
    root
        the root path to save the model
    name
        the name used to save the model
    l1_reg
        l1 regularization factor
    l2_reg
        l2 regularization factor
    nodes
        a list that contains the numbers of the nodes of hidden layers
    lrr_patience
        The patience for learning rate reduction
    ES_patience
        The patience for early stopping

    Returns
    -------
    model
        a trained fully connected neural network
    '''

    length = Y.shape[1]

    earlystopper = EarlyStopping(monitor=loss_name, patience=ES_patience, verbose=1)

    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor=loss_name, 
                                                patience=lrr_patience,
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=min_lr)

    model = FCNN((X.shape[1],), length, l1_reg = l1_reg, l2_reg = l2_reg,
                  dropout = dropout, nodes = nodes, seed = seed)
    model.compile(optimizer = 'adam', loss = loss, metrics = [loss])

    if (seed == None):
        shuffle = True
    else:
        shuffle = False
    model.fit_generator(generator = batch_generator(X,\
                                                  Y,\
                                                  np.arange(X.shape[0]),\
                                                  batch_size, shuffle),\
                      epochs = epoch,\
                      steps_per_epoch = X.shape[0] / batch_size,\
                      callbacks = [learning_rate_reduction, earlystopper])

    if(save):
        if not os.path.exists(root):
            os.makedirs(root)
        model.save(root + name + '.h5')
    
    return model

def WassersteinD(adata_ref: anndata.AnnData, adata_query: anndata.AnnData,
                 WD_cutoff: float, sparse: bool, root = 'WDs/', save = 'WDs_T1'):
    '''
    Calculate Wassertein distances of genes between two datasets

    Extract the shared genes that have a Wasserstein distance lower than `WD_cutoff`
     between the reference data and the query data
     Only the select genes will be kept in the reference and query Anndata objects

    Parameters
    ----------
    adata_ref
        Reference anndata object. Gene expression matrix should be the shape of (cell, gene).
        Spatial information should be stored in adata_ref.obsm['spatial'] in `np.array` format
    adata_query
        Query anndata object. Gene expression matrix should be the shape of (cell, gene).
    sparse
        if gene expression is saved in sparse format
    WD_cutoff
        genes with Wasserstein distance lower than the cutoff will be selected

    Returns
    -------
    JGs
        A list that contains the select genes
    WDs
        A table contains
    '''
    #adata_ref.var_names = adata_ref.var_names.str.upper()
    #adata_query.var_names = adata_query.var_names.str.upper()

    #adata_ref.var_names_make_unique()
    #adata_query.var_names_make_unique()

    JGs = sorted(list(set(adata_ref.var_names).intersection(set(adata_query.var_names))))

    WDs = pd.DataFrame(np.ones((len(JGs), 2)), columns=['Gene', 'Wasserstein_Distance'])
    WDs['Gene'] = JGs
    for i, gene in enumerate(JGs):
        if (sparse):
            gene_ref = adata_ref[:, gene].X.toarray().flatten()
            gene_query = adata_query[:, gene].X.toarray().flatten()
        else:
            gene_ref = adata_ref[:, gene].X.flatten()
            gene_query = adata_query[:, gene].X.flatten()
        WDs.iloc[i, 1] = wasserstein_distance(gene_ref, gene_query)
        print('Select Genes based on Wasserstein Distance')
        print('*' * 32)
        print(i, '/', WDs.shape[0])
        clear_output(wait=True)

    JGs = sorted(WDs[WDs['Wasserstein_Distance'] < WD_cutoff]['Gene'].tolist())

    if(save!=None):
        if not os.path.exists(root):
            os.makedirs(root)
        WDs.to_csv(root + save + '.csv', index = None)

    return JGs, WDs

def pp_Mapping(adata_ref:anndata.AnnData, adata_query:anndata.AnnData,
               JGs: list, sparse = True, WD_cutoff = None):

    '''
    Extract gene expression matrices sharing the same genes and the reference coordinates

    Extract the reference gene expression matrix and the query gene expression matrix
    that contain the shared genes, and the reference coordinates from anndata objects

    Parameters
    ----------
    adata_ref
        Reference anndata object. Gene expression matrix should be the shape of (cell, gene).
        Spatial information should be stored in adata_ref.obsm['spatial'] in `np.array` format
    adata_query
        Query anndata object. Gene expression matrix should be the shape of (cell, gene).
    JGs
        List of genes used for training
    sparse
        if gene expression is saved in sparse format
    WD_cutoff
        genes with Wasserstein distance lower than the cutoff will be selected

    Returns
    -------
    X_ref
        reference shared gene expression array
    X_query
        query shared gene exression array
    Y_ref
        reference spatial coordinates
    '''

    # Select genes
    if(JGs == None):
        if (WD_cutoff == None):
            #adata_ref.var_names = adata_ref.var_names.str.upper()
            #adata_query.var_names = adata_query.var_names.str.upper()

            adata_ref.var_names_make_unique()
            adata_query.var_names_make_unique()

            JGs = sorted(list(set(adata_ref.var_names).intersection(set(adata_query.var_names))))
        else:
            JGs, WDs = WassersteinD(adata_ref, adata_query, WD_cutoff, sparse)

    print('n of Referece Genes:', adata_ref.shape[1])
    print('n of Target Genes:', adata_query.shape[1])
    print('n of Selected Genes:', len(JGs))

    print(adata_ref[:, JGs].shape)
    print(adata_query[:, JGs].shape)

    if(sparse):
        X_ref = adata_ref[:, JGs].X.toarray()
        X_query = adata_query[:, JGs].X.toarray()
    else:
        X_ref = adata_ref[:, JGs].X
        X_query = adata_query[:, JGs].X
    Y_ref = adata_ref.obsm['spatial']

    return X_ref, X_query, Y_ref

def BatchPredict(Model, X, BatchSize = 60000):
    '''
    Predict batch by batch

    Parameters
    ----------
    Model
        A trained fully connected neural network
    X
        a numpy array with the shape of (cell, gene)
    BatchSize
        number of cells for one batch

    Returns
    -------
    Preds
        Predicted locations of X
    '''
    if(X.shape[0] > BatchSize):
        Preds = []
        NoB = int(np.ceil(X.shape[0] / BatchSize))
        for i in range(NoB):
            Pred = Model.predict(X[i*BatchSize:(i+1) * BatchSize, :])
            Preds.append(Pred)
        Preds = np.concatenate(Preds)
    else:
        Preds = Model.predict(X)

    return Preds

def ExtractXY(adata, sparse = True, polar = True):

    '''
    Preprocess the coordinates

    Parameters
    ----------
    adata
        Reference anndata object. Gene expression matrix should be the shape of (cell, gene).
        Spatial information should be stored in adata_ref.obsm['spatial'] in `np.array` format
    sparse
        If gene expression is saved in sparse format
    polar
        Transform cartesian coordinates to polar coordinates and conduct min-max normalization
        if True, otherwise apply min-max normalization to the cartesian coordinates.

    Returns
    -------
    X
        Reference gene expression array
    Y
        Transformed reference spatial coordinates
    Y_ref
        The original reference spatial coordinates
    RTheta_ref
        The original R and Theta values of the polar coordinate. None if polar is False.
    '''

    if(sparse):
        X = adata.X.toarray()
    else:
        X = adata.X
    Y_ref = adata.obsm['spatial']

    if(polar):
        RTheta_ref = pp.PolarTrans(Y_ref)
        Y = pp.MinMaxNorm(RTheta_ref, RTheta_ref)
    else:
        Y = pp.MinMaxNorm(Y_ref, Y_ref)
        RTheta_ref = None

    return X, Y, Y_ref, RTheta_ref

def Self_Mapping(adata: anndata.AnnData, sparse = True, model_path = None,
            root = 'Model_SI/', name = 'SI_Overall', l1_reg = 1e-5, l2_reg = 0, dropout = 0.05, epoch = 500,
            batch_size = 4096, nodes = [4096, 1024, 256, 64, 16, 4], lrr_patience = 20, ES_patience = 50,
            min_lr = 1e-5, save = False, polar = True, seed = None):

    '''
    Map ST beads to spatial locations. A fully connected neural network trained on all
    all ST beads will be applied to the same set of beads. A model will be trained and
    saved to `root+name+'.h5'` if model_path is None and save is True.
    The predicted coordinates will be saved in adata.obsm['spatial_self_mapping']

    Parameters
    ----------
    adata
        Reference anndata object. Gene expression matrix should be the shape of (cell, gene).
        Spatial information should be stored in adata_ref.obsm['spatial'] in `np.array` format
    model_path
        The path of a trained model. If not None, parameters for training will be ignored.
    root
        the root path to save the model
    name
        the name used to save the model
    l1_reg
        l1 regularization factor
    l2_reg
        l2 regularization factor
    nodes
        a list that contains the numbers of the nodes of hidden layers
    lrr_patience
        The patience for learning rate reduction
    min_lr
        minimum learning rate
    ES_patience
        The patience for early stopping

    Returns
    -------
    model
        a trained fully connected neural network

    '''

    #Extract values
    X, Y, Y_ref, RTheta_ref = ExtractXY(adata=adata, sparse = sparse, polar=polar)

    if(model_path != None):
        model = load_model(model_path, compile=False)
    else:
        model = Train(X=X, Y=Y, root = root, name = name, l1_reg = l1_reg,
                      l2_reg = l2_reg, dropout = dropout, epoch = epoch,
                      batch_size = batch_size, nodes = nodes, lrr_patience = lrr_patience,
                      ES_patience = ES_patience, min_lr = min_lr, save = save, seed = seed)
    Ref_pred_Y = BatchPredict(model, X)
    if(polar):
        Ref_pred_Y = pp.RePolarTrans(pp.ReMMNorm(RTheta_ref, Ref_pred_Y))
    else:
        Ref_pred_Y = pp.ReMMNorm(Y_ref, Ref_pred_Y)

    #Write prediction into adata
    adata.obsm['spatial_mapping'] = Ref_pred_Y

    return model

def Mapping(adata_ref, adata_query, sparse = True, model_path = None, WD_cutoff = None,
            root = 'Model_SI/', name = 'SI', l1_reg = 1e-5, l2_reg = 0, dropout = 0.05, epoch = 500,
            batch_size = 4096, nodes = [4096, 1024, 256, 64, 16, 4], lrr_patience = 20, ES_patience = 50,
            min_lr = 1e-5, save = False, polar = True, seed = None):

    '''
    Map single cells to spatial locations.
    A model will be trained and saved to `root+name+'.h5'` if model_path is None and save is True.
    The predicted coordinates will be saved in adata_query.obsm['spatial_mapping']

    Parameters
    ----------
    adata_ref
        Reference anndata object. Gene expression matrix should be the shape of (cell, gene).
        Spatial information should be stored in adata_ref.obsm['spatial'] in `np.array` format
    adata_query
        Query anndata object. Gene expression matrix should be the shape of (cell, gene).
    sparse
        if gene expression is saved in sparse format
    model_path
        The path of a trained model. If not None, parameters for training will be ignored.
    WD_cutoff
        genes with Wasserstein distance lower than the cutoff will be used for mapping
    root
        the root path to save the model
    name
        the name used to save the model
    l1_reg
        l1 regularization factor
    l2_reg
        l2 regularization factor
    nodes
        a list that contains the numbers of the nodes of hidden layers
    lrr_patience
        The patience for learning rate reduction
    min_lr
        minimum learning rate
    ES_patience
        The patience for early stopping

    Returns
    -------
    None

    '''

    #Extract values
    X_ref, X_query, Y_ref = pp_Mapping(adata_ref, adata_query,
                                       sparse = sparse, WD_cutoff = WD_cutoff)

    if(polar):
        RTheta_ref = pp.PolarTrans(Y_ref)
        Y = pp.MinMaxNorm(RTheta_ref, RTheta_ref)
    else:
        Y = pp.MinMaxNorm(Y_ref, Y_ref)

    if(model_path != None):
        model = load_model(model_path, compile=False)
    else:
        model = Train(X=X_ref, Y=Y, root = root, name = name, l1_reg = l1_reg,
                      l2_reg = l2_reg, dropout = dropout, epoch = epoch,
                      batch_size = batch_size, nodes = nodes, lrr_patience = lrr_patience,
                      ES_patience = ES_patience, min_lr = min_lr, save = save, seed = seed)
    Query_pred_Y = BatchPredict(model, X_query)
    if(polar):
        Query_pred_Y = pp.RePolarTrans(pp.ReMMNorm(RTheta_ref, Query_pred_Y))
    else:
        Query_pred_Y = pp.ReMMNorm(Y_ref, Query_pred_Y)

    #Write prediction into adata
    adata_query.obsm['spatial_mapping'] = Query_pred_Y


def Reconstruct_scST(adata_ref, adata_query, n_neighbors=1000,
                     dis_cutoff=15, n_layer_cell= [1, 4], cell_radius=5,
                     seed=2023):
    '''
    Reconstruct ST data at single cell resolution

    1 Finely map single cells to spatial locations.
    A model will be trained and saved to `root+name+'.h5'` if model_path is None and save is True.
    The predicted coordinates of single cells will be saved in adata_query.obsm['spatial_mapping']
    The predicted coordinates of beads will be saved in adata_ref.obsm['spatial_mapping']
    Fine mapping information will be saved in adata_ref.obs['FM'] and adata_query.obs['FM']. True if a cell/bead
    was mapped, otherwise False.

    2 Reconstruct ST data at single cell resolution
    adata_query.obs['Dis2CloestBead'] stores the distance between a cell and the ST bead closest to it

    Parameters
    ----------
    adata_ref
        Reference anndata object. Gene expression matrix should be the shape of (cell, gene).
        Spatial information should be stored in adata_ref.obsm['spatial'] in `np.array` format
    adata_query
        Query anndata object. Gene expression matrix should be the shape of (cell, gene).
    n_neighbors
        Number of the nearest neighbors of a bead or cell. This parameter is for
        the KNN algorithm
    dis_cutoff
        Limit for the distance between a single cell and a ST bead. In the process
         of fine mapping. Only the cells within the cutoff will be retained.
    n_layer_cell
        Number of cells in layers for sampling single cells for a ST bead.
    cell_radius
        Radius of a cell. For example, n_layer_cell=[1, 4] and cell_radius=5 means sampling 1 cell
         from cells within 5 to a bead and at most 4 cells from cells between 5 and 15 to the bead.
         It is at most 4 cells because a cell can be sampled more than once to deal with the case
         that a bead has fewer cells than the user specified.
    seed
        seed for reconstructing the single-cell ST data
    '''

    # Extract location prediction
    Ref_pred_Y = adata_ref.obsm['spatial_mapping']
    Query_pred_Y = adata_query.obsm['spatial_mapping']

    ##Fine Mapping
    # Merge Prediction
    Query_pred_Y = pd.DataFrame(Query_pred_Y, columns=['x_transfer', 'y_transfer'])
    Query_pred_Y['source'] = 'SC'
    Ref_pred_Y = pd.DataFrame(Ref_pred_Y, columns=['x_transfer', 'y_transfer'])
    Ref_pred_Y['source'] = 'ST'
    transfer_merge = pd.concat([Ref_pred_Y, Query_pred_Y])

    n_ST = Ref_pred_Y.shape[0]
    coords = transfer_merge[['x_transfer', 'y_transfer']]
    neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=dis_cutoff)
    neigh.fit(coords)
    dis, neighbors = neigh.kneighbors(coords, n_neighbors)

    # Correct overlapping beads/cells
    neighbors[:, 0] = list(range(neighbors.shape[0]))

    # Extract Single Cell Info
    dis_sc = dis[n_ST:]
    neighbors_sc = neighbors[n_ST:]

    # Remove single cells
    # Remove beads that don't have cells/beads around within cutoff distance
    ST_mask1 = dis[:n_ST, 1:].min(axis=1) < dis_cutoff
    dis_st = dis[:n_ST][ST_mask1]
    neighbors_st = neighbors[:n_ST][ST_mask1]

    # Remove beads that don't have single cells around within cutoff distance
    neighbors_st[dis_st > dis_cutoff] = -1
    ST_mask2 = (neighbors_st >= n_ST).sum(axis=1) > 0
    dis_st = dis_st[ST_mask2]
    neighbors_st = neighbors_st[ST_mask2]

    # Extract mapped indices of single cells and ST beads
    MappedIndices = np.unique(neighbors_st)
    MappedIndices = MappedIndices[MappedIndices != -1]

    MappedSTInd = neighbors_st[:, 0]
    MappedSCInd = MappedIndices[MappedIndices >= n_ST] - n_ST

    # Write info into adata
    adata_ref.obs['FM'] = False
    ref_FM_ind = np.where(adata_ref.obs.columns == 'FM')[0][0]
    adata_ref.obs.iloc[MappedSTInd, ref_FM_ind] = True

    # Write fine mapping information of single cells
    adata_query.obs['FM'] = False
    query_FM_ind = np.where(adata_query.obs.columns == 'FM')[0][0]
    adata_query.obs.iloc[MappedSCInd, query_FM_ind] = True

    ## Reconstruct ST data at single cell resolution
    # Calculate cells' distance to the closest ST bead
    dis_sc[neighbors_sc >= n_ST] = 1e6
    dis_sc[:, 0] = 1e6
    adata_query.obs['Dis2ClosestBead'] = dis_sc.min(axis=1)

    #Label closest ST beads of single cells
    adata_query.obs['ClosestBead_order'] = neighbors_sc[range(neighbors_sc.shape[0]),
                                                        dis_sc.argmin(axis=1)]
    adata_query.obs['ClosestBead_name'] = adata_query.obs['ClosestBead_order'].apply(lambda x: adata_ref.obs_names[x] if x < n_ST else 'NA')

    #Label closest single cells of ST beads
    # ClosestSC_list = adata_query.groupby('ClosestBead_name', as_index = False)['Dis2ClosestBead'].idxmin()['Dis2ClosestBead'].tolist()
    index_name = adata_query.obs.reset_index().columns[0]
    ClosestSC_list = adata_query.obs.reset_index().groupby('ClosestBead_name', as_index=False).agg(
        {'Dis2ClosestBead': 'min', index_name: 'first'})[index_name].tolist()
    adata_query.obs['ClosestSC'] = False
    adata_query.obs.loc[ClosestSC_list, 'ClosestSC'] = True

    adata_query.obs['Recon_scST'] = False
    adata_query.obs['Recon_scST_layer'] = -1
    for layer, n_cell in enumerate(n_layer_cell):

        if (layer == 0):
            lower_bound = adata_query.obs['Dis2ClosestBead'] > -0.01
            upper_bound = adata_query.obs['Dis2ClosestBead'] < ((layer + 1) * cell_radius)
        else:
            lower_bound = (adata_query.obs['Dis2ClosestBead'] > ((2 * layer - 1) * cell_radius)) | \
                          (adata_query.obs['Dis2ClosestBead'] == ((2 * layer - 1) * cell_radius))
            upper_bound = adata_query.obs['Dis2ClosestBead'] < ((2 * layer + 1) * cell_radius)

        temp = adata_query.obs[lower_bound & upper_bound & (~adata_query.obs['Recon_scST'])]
        temp = temp.groupby('ClosestBead_name').sample(n=n_cell, replace = True,
                                                       random_state=seed).index
        temp = pd.Series(temp).tolist()

        adata_query.obs.loc[temp, 'Recon_scST'] = True
        adata_query.obs.loc[temp, 'Recon_scST_layer'] = layer

def FineMapping(adata_ref, adata_query, sparse = True, model_path = None, WD_cutoff = None, JGs = None,
                root = 'Model_SI/', name = 'SI', l1_reg = 1e-5, l2_reg = 0, dropout = 0.05, epoch = 500,
                batch_size = 4096, nodes = [4096, 1024, 256, 64, 16, 4], lrr_patience = 20, ES_patience = 50,
                min_lr = 1e-5, save = True, polar = True, FM = True, n_layer_cell = [1, 4],
                cell_radius = 5, n_neighbors = 1000, dis_cutoff = 15, seed = 2023):

    '''
    Finely map single cells to spatial locations and Reconstruct ST data at single cell resolution

    1. Finely map single cells to spatial locations.
    A model will be trained and saved to `root+name+'.h5'` if model_path is None and save is True.
    The predicted coordinates of single cells will be saved in adata_query.obsm['spatial_mapping']
    The predicted coordinates of beads will be saved in adata_ref.obsm['spatial_mapping']
    Fine mapping information will be saved in adata_ref.obs['FM'] and adata_query.obs['FM']. True if a cell/bead
    was mapped, otherwise False.

    2. Reconstruct ST data at single cell resolution
    adata_query.obs['Dis2CloestBead'] stores the distance between a cell and the ST bead closest to it
    adata_query.obs['Recon_scST'] marks the cells that are used to reconstruct single-cell ST data
    adata_query.obs['Recon_scST_layer'] stores which reconstruction layer of cells. -1 means
    a cell is not assigned to any ST bead. 0 means a cell is among the closest cells to a ST bead.
    The greater the layer number, the further a cell is to the center of a ST bead.

    Parameters
    ----------
    adata_ref
        Reference anndata object. Gene expression matrix should be the shape of (cell, gene).
        Spatial information should be stored in adata_ref.obsm['spatial'] in `np.array` format
    adata_query
        Query anndata object. Gene expression matrix should be the shape of (cell, gene).
    sparse
        if gene expression is saved in sparse format
    model_path
        The path of a trained model. If not None, parameters for training will be ignored.
    WD_cutoff
        genes with Wasserstein distance lower than the cutoff will be used for mapping
    JGs
        List of genes used for training
    root
        the root path to save the model
    name
        the name used to save the model
    l1_reg
        l1 regularization factor
    l2_reg
        l2 regularization factor
    nodes
        a list that contains the numbers of the nodes of hidden layers
    lrr_patience
        The patience for learning rate reduction
    min_lr
        minimum learning rate
    ES_patience
        The patience for early stopping
    n_neighbors
        Number of the nearest neighbors of a bead or cell. This parameter is for
        the KNN algorithm
    dis_cutoff
        Limit for the distance between a single cell and a ST bead. In the process
         of fine mapping. Only the cells within the cutoff will be retained.
    FM
        Perform fine mapping and reconstruct ST data at single cell resolution if True
    n_layer_cell
        Number of cells in layers for sampling single cells for a ST bead.
    cell_radius
        Radius of a cell. For example, n_layer_cell=[1, 4] and cell_radius=5 means sampling 1 cell
         from cells within 5 to a bead and at most 4 cells from cells between 5 and 15 to the bead.
         It is at most 4 cells because a cell can be sampled more than once to deal with the case
         that a bead has fewer cells than the user specified.
    seed
        seed for reconstructing the single-cell ST data
    '''

    # Extract values
    X_ref, X_query, Y_ref = pp_Mapping(adata_ref, adata_query, JGs = JGs,
                                       sparse = sparse, WD_cutoff = WD_cutoff)

    if(polar):
        RTheta_ref = pp.PolarTrans(Y_ref)
        Y = pp.MinMaxNorm(RTheta_ref, RTheta_ref)
    else:
        Y = pp.MinMaxNorm(Y_ref, Y_ref)

    if(model_path != None):
        model = load_model(model_path, compile=False)
    else:
        model = Train(X=X_ref, Y=Y, root = root, name = name, l1_reg = l1_reg, l2_reg = l2_reg,
                      dropout = dropout, epoch = epoch, batch_size = batch_size, nodes = nodes,
                      lrr_patience = lrr_patience, ES_patience = ES_patience, min_lr = min_lr,
                      save = save, seed = seed)
    Query_pred_Y = BatchPredict(model, X_query)
    Ref_pred_Y = BatchPredict(model, X_ref)
    if(polar):
        Query_pred_Y = pp.RePolarTrans(pp.ReMMNorm(RTheta_ref, Query_pred_Y))
        Ref_pred_Y = pp.RePolarTrans(pp.ReMMNorm(RTheta_ref, Ref_pred_Y))
    else:
        Query_pred_Y = pp.ReMMNorm(Y_ref, Query_pred_Y)
        Ref_pred_Y = pp.ReMMNorm(Y_ref, Ref_pred_Y)

    # Write info into adata
    adata_ref.obsm['spatial_mapping'] = Ref_pred_Y
    adata_query.obsm['spatial_mapping'] = Query_pred_Y

    if(FM):
        Reconstruct_scST(adata_ref, adata_query, n_neighbors = n_neighbors,
                    dis_cutoff = dis_cutoff, n_layer_cell = n_layer_cell,
                    cell_radius=cell_radius, seed = seed)

def NRD_CT_preprocess(adata_ref, adata_query, n_neighbors=1000, dis_cutoff=15):
    '''
    Apply KNN algorithm to obtain the single cell neighbors of ST beads

    Parameters
    ----------
    adata_ref
        Reference anndata object. Gene expression matrix should be the shape of (cell, gene).
        The predicted coordinates of beads should be stored in adata_ref.obsm['spatial_mapping']
    adata_query
        Query anndata object. Gene expression matrix should be the shape of (cell, gene).
        The predicted coordinates of single cells should be stored in adata_query.obsm['spatial_mapping'].
    n_neighbors
        Number of the nearest neighbors of a bead or cell. This parameter is for the KNN algorithm
    dis_cutoff
        Maximum distance between a single cell and a ST bead. Only the cells within the cutoff
         will be retained for further analysis.

    Returns
    -------
    neighbors_st
        A matrix of single cell neighbors of ST beads with the shape of (n_beads, n_neighbors)
    dis_st
        The distance matrix of the neighbor matrix
    '''

    # Extract location prediction
    Ref_pred_Y = adata_ref.obsm['spatial_mapping']
    Query_pred_Y = adata_query.obsm['spatial_mapping']

    ##Fine Mapping
    # Merge Prediction
    Query_pred_Y = pd.DataFrame(Query_pred_Y, columns=['x_transfer', 'y_transfer'])
    Query_pred_Y['source'] = 'SC'
    Ref_pred_Y = pd.DataFrame(Ref_pred_Y, columns=['x_transfer', 'y_transfer'])
    Ref_pred_Y['source'] = 'ST'
    transfer_merge = pd.concat([Ref_pred_Y, Query_pred_Y])

    n_ST = Ref_pred_Y.shape[0]
    coords = transfer_merge[['x_transfer', 'y_transfer']]
    neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=dis_cutoff)
    neigh.fit(coords)
    dis, neighbors = neigh.kneighbors(coords, n_neighbors)

    # Correct overlapping beads/cells
    neighbors[:, 0] = list(range(neighbors.shape[0]))

    # Remove single cells
    # Remove beads that don't have cells/beads around within cutoff distance
    ST_mask1 = dis[:n_ST, 1:].min(axis=1) < dis_cutoff
    dis_st = dis[:n_ST][ST_mask1]
    neighbors_st = neighbors[:n_ST][ST_mask1]

    # Remove beads that don't have single cells around within cutoff distance
    neighbors_st[dis_st > dis_cutoff] = -1
    ST_mask2 = (neighbors_st >= n_ST).sum(axis=1) > 0
    dis_st = dis_st[ST_mask2]
    neighbors_st = neighbors_st[ST_mask2]

    return neighbors_st, dis_st

def NRD_weight(neighbors, dis, adata_ref, adata_query, ct_name=None, weight_constant = 1):

    '''
    Calculate the weights of nearby single cells for ST beads based on the fine mapping result

    Parameters
    ---------
    neighbors
        Matrix of single cell neighbors of ST beads with the shape of (n_beads, n_neighbors)
    dis
        corresponding distance matrix of the neighbor matrix
    adata_ref
        Reference anndata object. Gene expression matrix should be the shape of (cell, gene).
        Predicted locations should be stored in adata_ref.obs[['x_transfer', 'y_transfer']]
    adata_query
        Query anndata object. Gene expression matrix should be the shape of (cell, gene).
        Cell type annotation should be stored in adata_query.obs[ct_name]
        Predicted locations should be stored in adata_query.obs[['x_transfer', 'y_transfer']]
    weight_constant
        A constant added up to the distance between a cell and a ST voxel when calculating
        the weight of the cell to the ST voxel.

    Returns
    -------
    df_neighbor
        A dataframe that contains the cell type information and normalized reciprocal distance
         of mapped ST beads and single cells

    '''

    n_ST = adata_ref.shape[0]

    df_neighbor = np.concatenate([np.repeat(neighbors[:, 0], neighbors.shape[1] - 1).reshape((-1, 1)),
                                  neighbors[:, 1:].flatten().reshape((-1, 1)),
                                  dis[:, 1:].flatten().reshape((-1, 1))], axis=1)
    df_neighbor = pd.DataFrame(df_neighbor, columns=['center', 'neighbor', 'dis'])
    # Select single cell neighbors for ST beads
    # Remove cell pairs whose distance are more than the cutoff distance
    ##(ID was labelled as -1 in the fine mapping step)
    df_neighbor = df_neighbor[df_neighbor['neighbor'] >= n_ST].reset_index(drop=True)
    df_neighbor[['center', 'neighbor']] = df_neighbor[['center', 'neighbor']].astype(int)
    if(ct_name!=None):
        df_neighbor['SCT'] = adata_query.obs.iloc[df_neighbor['neighbor'] - n_ST][ct_name].tolist()

    df_neighbor['weight'] = 1 / (df_neighbor['dis'] + weight_constant)

    return df_neighbor


def NRD_CT(neighbors, dis, adata_ref, adata_query, ct_name='simp_name', weight_constant = 1,
           exclude_CTs=['nan']):

    '''
    Normalized Reciprocal Distance

    Calculate the proportion of cell types for ST beads based on the NRD_weight output
    Predicted cell type proportion will be saved in adata_ref.obs.

    Parameters
    ---------
    neighbors
        Matrix of single cell neighbors of ST beads with the shape of (n_beads, n_neighbors)
    dis
        corresponding distance matrix of the neighbor matrix
    adata_ref
        Reference anndata object. Gene expression matrix should be the shape of (cell, gene).
        Spatial information should be stored in adata_ref.obs[[x_name, y_name]]
    adata_query
        Query anndata object. Gene expression matrix should be the shape of (cell, gene).
        Cell type annotation should be stored in adata_query.obs[ct_name]
        Predicted locations should be stored in adata_query.obs[['x_transfer', 'y_transfer']]
    dis_min
        Distance cutoff that determines if a ST bead and a single cell is paired
    exclude_CTs
        A list that contains cell types to be excluded
    weight_constant
        A constant added up to the distance between a cell and a ST voxel when calculating
        the weight of the cell to the ST voxel.

    Returns
    -------
    None

    '''

    df_neighbor = NRD_weight(neighbors, dis, adata_ref, adata_query,
                             ct_name=ct_name, weight_constant=weight_constant)

    # Exclude CTs
    if((exclude_CTs != None)&(ct_name!=None)):
        select = df_neighbor['SCT'].apply(lambda x: x not in exclude_CTs)
        df_neighbor = df_neighbor[select]

    # Calculate normalized reciprocal distance for each cell type
    CT_NRD = df_neighbor[['center', 'SCT', 'weight']].groupby(['center', 'SCT']).max()
    CT_NRD = CT_NRD / CT_NRD.groupby(['center']).sum()

    index = df_neighbor['center'].unique()
    columns = df_neighbor['SCT'].unique()
    CT_NRD_df = pd.DataFrame(np.zeros((len(index),
                                       len(columns))),
                             index=index,
                             columns=columns)

    for i in range(CT_NRD.shape[0]):
        prop = CT_NRD.iloc[i, 0]
        center = CT_NRD.index[i][0]
        CT = CT_NRD.index[i][1]

        CT_NRD_df.loc[center, CT] = prop

    adata_ref.obs['CT_NRD'] = False
    ref_CT_NRD_ind = np.where(adata_ref.obs.columns == 'CT_NRD')[0][0]
    adata_ref.obs.iloc[CT_NRD_df.index, ref_CT_NRD_ind] = True

    CT_NRD_df.index = adata_ref.obs.index[CT_NRD_df.index]
    adata_ref.obs = adata_ref.obs.merge(CT_NRD_df, how='left', left_index=True, right_index=True)


def NRD_impute(neighbors, dis, adata_ref, adata_query, ct_name=None,
               weight_constant = 1, exclude_CTs=None):

    '''
    Reconstruct the gene expression profile of ST beads based on the NRD_weight output

    Predicted cell type proportion will be saved in adata_ref.obs.

    Parameters
    ---------
    neighbors
        Matrix of single cell neighbors of ST beads with the shape of (n_beads, n_neighbors)
    dis
        corresponding distance matrix of the neighbor matrix
    adata_ref
        Reference anndata object. Gene expression matrix should be the shape of (cell, gene).
        Predicted coordinates should be stored in adata_ref.obs[['x_transfer', 'y_transfer']]
    adata_query
        Query anndata object. Gene expression matrix should be the shape of (cell, gene).
        Cell type annotation should be stored in adata_query.obs[ct_name]
        Predicted locations should be stored in adata_query.obs[['x_transfer', 'y_transfer']]
    dis_min
        Distance cutoff that determines if a ST bead and a single cell is paired
    exclude_CTs
        A list that contains cell types to be excluded

    Returns
    -------
    adata_impute
        An Anndata object that stores the reconstructed gene expression profile of ST beads

    '''
    df_neighbor = NRD_weight(neighbors, dis, adata_ref, adata_query,
                             ct_name=ct_name, weight_constant = weight_constant)

    if((exclude_CTs != None)&(ct_name!=None)):
        select = df_neighbor['SCT'].apply(lambda x: x not in exclude_CTs)
        df_neighbor = df_neighbor[select]

    CT_NRD = df_neighbor[['center', 'neighbor', 'weight']].groupby(['center', 'neighbor']).sum()
    CT_NRD = CT_NRD / CT_NRD.groupby(['center']).sum()

    n_ST = adata_ref.shape[0]
    STBeads_index = CT_NRD.index.get_level_values(0).unique().tolist()
    X_impute = np.zeros((len(STBeads_index), adata_query.shape[1]))

    i = 0
    for STB_index in STBeads_index:
        t = (adata_query[CT_NRD.loc[STB_index].index - n_ST].X.toarray() * CT_NRD.loc[STB_index].values).sum(axis=0)
        X_impute[i] = t
        i += 1

    adata_impute = anndata.AnnData(X=X_impute,
                                   obs=adata_ref.obs.iloc[STBeads_index])
    adata_impute.obsm['spatial'] = adata_ref.obsm['spatial'][STBeads_index]
    adata_impute.var_names = adata_query.var.index

    return adata_impute

def LR(input_shape: tuple, out_shape: int,
        l1_reg = 1e-45, l2_reg = 0,
        dropout = 0.05):
    '''
    Construct logistic regression model

    Parameters
    ---------
    input_shape
        a tuple with the shape of (number of genes, )
    out_shape
        the dimension of coordinates

    Returns
    -------
    model
        a logistical regression model

    '''

    X_input = Input(input_shape)
    
    X = X_input
    if(dropout != 0):
        X = Dropout(rate = dropout)(X)
    X = Dense(out_shape, kernel_initializer = 'glorot_normal',
              activation = 'sigmoid',
              kernel_regularizer = l1_l2(l1_reg, l2_reg))(X)
    
    model = Model(inputs = X_input, outputs = X, name = 'LR')
    
    return model
    
def CV_LR(X:np.array, Y:np.array, train_indices:list, test_indices:list,
          l1_reg = 1e-5, l2_reg = 0, dropout = 0.05,
          epoch = 20, batch_size = 4096):
    '''
    Cross-validation using logistic regression

    Parameters
    ---------
    X
        a numpy array with the shape of (cell, gene)
    Y
        a numpy array with the shape of (cell, dimension), where dimension is
         the dimension of spatial inforamtion
    train_indices
        a list contains mutiple lists of the index of cells for training
    test_indices
        a list contains mutiple lists of the index of cells for test. The train_indices
         and test_indices are paired based on the order of indices list.
    l1_reg
        l1 regularization factor

    Returns
    -------
    histories
        training history
    train_preds
        predicted locations of cells for training, which corresponds to train_indices.
    test_preds
        predicted locations of cells for test, which corresponds to test_indices

    '''
    length = Y.shape[1]

    earlystopper = EarlyStopping(patience=50, verbose=1)

    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_rmse', 
                                                patience=3, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=1e-8)

    histories = []
    train_preds = []
    test_preds = []

    i = 0
    while(i < 5):
        
        print(i)
        
        cv_train = train_indices[i].copy()
        cv_test = test_indices[i].copy()

        X_test = X[cv_test, :].copy()
        X_train = X[cv_train, :] 
        
        model = LR((X.shape[1],), length, l1_reg = l1_reg, l2_reg = l2_reg, dropout = dropout)
        model.compile(optimizer = 'adam', loss = rmse, metrics = [rmse])

        history = model.fit_generator(generator = batch_generator(X,\
                                                        Y,\
                                                        cv_train,\
                                                        batch_size, True),\
                            epochs = epoch,\
                            steps_per_epoch = len(cv_train) / batch_size,\
                            validation_data = (X_test, Y[cv_test, :]),\
                            callbacks = [learning_rate_reduction, earlystopper])

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        if(np.isnan(train_pred).sum()>0):
            continue
        
        train_preds.append(train_pred)
        test_preds.append(test_pred)
        histories.append(history)
        gc.collect()
        
        i += 1
        
    return histories, train_preds, test_preds
    
def Train_transfer(adata, root, model_root, sparse = True, polar = True, CT = 'A',
                   lrr_patience = 20, ES_patience = 50, min_lr = 1e-5,
                   epoch = 500, batch_size = 4096, NLFT = 6, subLayer = False):

    '''
    Finetune location prediction model (FCNN) on a specific cell type

    A model will be trained and saved to `root + 'SI_' + CT + '.h5'`
    The predicted coordinates of single cells will be saved in adata_query.obsm['spatial_mapping']
    The predicted coordinates of beads will be saved in adata_ref.obs['spatial_mapping']
    Fine mapping information will be saved in adata_ref.obs['FM'] and adata_query.obs['FM']. True if a cell/bead
    was mapped, otherwise False.

    Parameters
    ----------
    adata
        An anndata object that contains the gene expression matrix of the target type of cells.
        The gene expression matrix should be the shape of (cell, gene).
        Spatial information should be stored in adata_ref.obsm['spatial'] in `np.array` format
    sparse
        True if gene expression is saved in sparse format, otherwise False
    model_root
        The path of a trained model.
    root
        the root path to save the model
    name
        the name used to save the model
    lrr_patience
        The patience for learning rate reduction
    min_lr
        minimum learning rate
    ES_patience
        The patience for early stopping
    sublayer
        whether to finetune part of the neural network
    NLFT
        the number of layers that are finetuned. Layers are
        counted from the last layer to the first layer

    Returns
    -------
    model
        The finetuned FCNN
    '''

    # Extract values
    X, Y, Y_ref, RTheta_ref = ExtractXY(adata=adata, sparse=sparse, polar=polar)

    earlystopper = EarlyStopping(monitor=loss_name, patience=ES_patience, verbose=1)

    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor=loss_name, 
                                                patience=lrr_patience,
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=min_lr)

    model = load_model(model_root, compile = False)
    if(subLayer):
        for layer in model.layers[:-NLFT]:
            layer.trainable = False
    model.compile(optimizer = 'adam', loss = loss, metrics = [loss])
    model.fit_generator(generator = batch_generator(X,\
                                                    Y,\
                                                    np.arange(X.shape[0]),\
                                                    batch_size, True),\
                        epochs = epoch,\
                        steps_per_epoch = X.shape[0] / batch_size,\
                        callbacks = [learning_rate_reduction, earlystopper])
    model.save(root + 'SI_' + CT + '.h5')

    return model
    
def SaveValidation(history, CV:bool, name:str):
    '''
    Save the FCNN or LR training history to 'log/training_log_' + name + '.pickle',

    Parameters
    ----------
    history
        training history(ies) returned by keras fit or fit_generate function
    CV
        True if history is a list that contrain multiple histories from Cross-validation

    Returns
    -------
    None

    '''

    if not os.path.exists('log/'):
        os.makedirs('log/')
    
    if(CV):
        histories = history
        
        with open('log/training_log_' + name + '.pickle', 'wb') as handle:
            pickle.dump([history.history for history in histories], handle)

        with open('log/training_log_' + name + '.pickle', 'rb') as handle:
            histories = pickle.load(handle)

        accuracy = []

        for history in histories:
            accuracy.append(history['val_' + loss_name][-1])

        print('Validation ' + loss_name)
        print(accuracy)
        print(np.mean(accuracy))
        
    else:
        with open('log/training_log_' + name + '.pickle', 'wb') as handle:
            pickle.dump(history, handle)

        with open('log/training_log_' + name + '.pickle', 'rb') as handle:
            history = pickle.load(handle)
            
        print(history.history['val_mse'][-1])
        
def CheckAccuracy(name:str, item_name = 'rmse'):
    '''
    Check the accuracies and mean accuracy of cross-validation

     Read the FCNN or LR training histories saved in 'log/training_log_' + name + '.pickle'
     and Output accuracies and mean accuracy of cross-validation

     Parameters
     ----------
     name
     item_name
        loss name saved in the history files

     Returns
     -------
     None

     '''
    with open('log/training_log_' + name + '.pickle', 'rb') as handle:
        histories = pickle.load(handle)

        accuracy = []

        for history in histories:
            accuracy.append(history['val_' + item_name][-1])

        print(name)
        print('Validation', item_name)
        print(accuracy)
        print(np.mean(accuracy))

