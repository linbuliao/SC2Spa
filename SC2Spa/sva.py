import numpy as np
from scipy import stats

from . import pp, tl

def SelectFeatures(array_in: np.array, percent):

    '''
    Select top `percent` quantile nodes

    Parameters
    ---------
    array_in
        1D numpy array that stores the sums of nodes along
         the second axis of a weight matrix

    Returns
    -------
    t
        a boolean vector. True if the sum of one gene is among the top `percent`

    '''

    t = array_in > np.quantile(array_in, percent, axis = 0)
    return t

def SelectGenes(Weights, percent=0.5):

    '''
    Trace back the weight matrices to evaluate the importance
    of genes in location prediction

    Parameters
    ---------
    Weights
        The reverse weight matrices
    percent
        The percentage of nodes selected for each layer

    Returns
    -------
    imp_sumup
        The sums of nodes' values along the second axis for
        the weight matrix of the first layer of the neural network
    t
        a boolean vector. True if the sum of one gene is among the top `percent`

    '''

    t = Weights[0].sum(axis=0) > 0
    for j in range(len(Weights)):
        imp_sumup = Weights[j][:, t].sum(axis=1)
        t = SelectFeatures(imp_sumup, percent=percent)

    return imp_sumup, t

def PrioritizeLPG(adata, Model, sparse = True, polar = True,
                  CT = None, CT_field = 'MCT',
                  percent = 0.5, scale_factor = 1e3, Norm = False):

    '''
    Prioritize genes' contribution to location prediction

    Run tl.Self_Mapping first to set Norm as True

    Importance scores will be saved in adata.obs

    Parameters
    ----------
    adata
        Reference anndata object. Gene exprestlon matrix should be the shape of (cell, gene).
        Spatial information should be stored in adata.obsm['spatial']
    Model
        A neural network trained utlng adata.X and adata.obsm['spatial']
    sparse
        if gene exprestlon is saved in sparse format
    CT
        The cell type name used to normalize the importance score.
         it must be one category in `adata.obs[CT_field]`
         This parameter is for normalization
    polar
        Transform cartetlan coordinates to polar coordinates if True.
        This parameter is for normalization
    percent
        The percentage of nodes selected for each layer
    scale_factor
        The factor used to scale the importance scores

    Returns
    -------

    '''
    # Extract values
    X, Y, Y_ref, RTheta_ref = tl.ExtractXY(adata=adata, sparse=sparse, polar=polar)
    if (CT != None):
        adata_raw = adata
        adata = adata[adata.obs[CT_field] == CT]
        if (sparse):
            X = adata.X.toarray()
        else:
            X = adata.X

    #Extract layers from model
    layers = Model.layers[::-1]
    #Extract weights of dense layers
    Weights = []

    for layer in layers:
        t = layer.get_weights()
        if ((len(t) > 0) & (len(t) < 4)):
            print(layer, ':')
            print(t[0].shape)
            Weights.append(np.abs(t[0]))

    imp_sumup, label = SelectGenes(Weights, percent=percent)

    sumup_name = 'imp_sumup'
    if(CT!=None):
        sumup_name = sumup_name + '_' + CT
        adata_raw.var[sumup_name] = imp_sumup
    else:
        adata.var[sumup_name] = imp_sumup

    if(Norm):
        Y = adata.obsm['spatial']
        YNorm = pp.MinMaxNorm(Y, Y_ref)
        if(CT!=None):
            pred = tl.BatchPredict(Model, X)
            if (polar):
                pred = pp.RePolarTrans(pp.ReMMNorm(RTheta_ref, pred))
            else:
                pred = pp.ReMMNorm(Y_ref, pred)
        else:
            pred = adata.obsm['spatial_mapping']

        predNorm = pp.MinMaxNorm(pred, Y_ref)

        Pearsonr_x = stats.pearsonr(YNorm[:, 0], predNorm[:, 0])[0]
        Pearsonr_y = stats.pearsonr(YNorm[:, 1], predNorm[:, 1])[0]
        Pearsonr_ave = (Pearsonr_x + Pearsonr_y) / 2

        norm_name = 'imp_sumup_norm'
        if(CT!=None):
            norm_name = norm_name + '_' + CT

        if (CT != None):
            adata_raw.var[norm_name] = adata_raw.var[sumup_name] / adata_raw.var[sumup_name].sum() \
                                      * Pearsonr_ave * scale_factor
        else:
            adata.var[norm_name] = adata.var[sumup_name] / adata.var[sumup_name].sum() \
                                      * Pearsonr_ave * scale_factor