import os
from time import time

import anndata as ad
import scanpy as sc
from SC2Spa import SI, PP, Vis, SVA

import pandas as pd

from numpy.random import seed
from tensorflow.random import set_seed
import tensorflow as tf


import SC2Spa
import wget

def test_mapping():

    print(SC2Spa.__all__)
    print('*'*20)
    print('Version:', SC2Spa.__version__)
    
    if not os.path.exists('Dataset'):
        os.makedirs('Dataset')

    wget.download('https://figshare.com/ndownloader/files/38736651', out = 'Dataset/AdataMH1.h5ad')
    wget.download('https://figshare.com/ndownloader/files/38738136', out = 'Dataset/AMB_HC.h5ad')
    wget.download('https://figshare.com/ndownloader/files/38756529', out = 'Dataset/ssHippo_RCTD.csv')
    
    if not os.path.exists('tutorial1'):
        os.makedirs('tutorial1')
    #%cd tutorial1
    
    #Load
    adata_ref = ad.read_h5ad('Dataset/AdataMH1.h5ad')
    adata_query = ad.read_h5ad('Dataset/AMB_HC.h5ad')
    
    adata_ref.var_names = adata_ref.var_names.str.upper()
    adata_query.var_names = adata_query.var_names.str.upper()
    
    adata_ref.var_names_make_unique()
    adata_query.var_names_make_unique()
    
    #Normalize
    sc.pp.normalize_total(adata_ref, target_sum=1e4)
    sc.pp.log1p(adata_ref)
    sc.pp.normalize_total(adata_query, target_sum=1e4)
    sc.pp.log1p(adata_query)
    
    #Load annotation
    Anno = pd.read_csv('Dataset/ssHippo_RCTD.csv', index_col = 0)
    Anno['MCT'] = 't'
    index1 = Anno.index[(Anno['celltype_1'] == Anno['celltype_2'])]
    Anno['MCT'][index1] = Anno['celltype_1'][index1]
    index2 = Anno.index[(Anno['celltype_1'] != Anno['celltype_2'])]
    Anno['MCT'][index2] = (Anno['celltype_1'][index2] + '_' + Anno['celltype_2'][index2]).apply(lambda x: '_'.join(sorted(set(x.split('_')))))
    adata_ref.obs = adata_ref.obs.merge(Anno, left_index = True, right_index = True, how = 'left')
    
    adata_ref.obsm['spatial'] = adata_ref.obs[['xcoord', 'ycoord']].values
    
    
    
    adata_query.obs['common_name'] = adata_query.obs['common_name'].str.replace('?', '')
    adata_query.obs['simp_name'] = adata_query.obs['common_name'].str.split('.', expand = True)[0].str.split(',', expand = True)[0].str.split(' \(', expand = True)[0].str.replace('cortexm', 'cortex').replace('Medial entorrhinal cortex', 'Medial entorhinal cortex')
    
    WD_cutoff = 0.4
    root = 'tutorial1/'
    save = 'WDs_T2'
    
    # Download the precalculated Wasserstain distances between the scRNA-seq and ST datasets.
    wget.download('https://figshare.com/ndownloader/files/38938874', out = 'tutorial1/WDs_T2.csv')
    
    WDs = pd.read_csv(root + save + '.csv')
    JGs = sorted(WDs[WDs['Wasserstein_Distance'] < WD_cutoff]['Gene'].tolist())
    
    #Download Pretrained Model
    wget.download('https://figshare.com/ndownloader/files/38938871', out = 'tutorial1/SI_T2_WD.h5')
    
    #Set random generator seed
    seed_num = 2023
    seed(seed_num)
    set_seed(seed_num)
    tf.keras.utils.set_random_seed(seed_num)
    
    '''
    Finely map single cells to spatial locations.
    A model will be trained and saved to `root+name+'.h5'` if model_path is None and save is True.
    The predicted coordinates of single cells will be saved in adata_query.obsm['spatial_mapping']
    The predicted coordinates of beads will be saved in adata_ref.obsm['spatial_mapping']
    Fine mapping information will be saved in adata_ref.obs['FM'] and adata_query.obs['FM']. True if a cell/bead
    was mapped, otherwise False.
    '''
    
    SI.FineMapping(adata_ref, adata_query, JGs = JGs, sparse =True, model_path = 'tutorial1/SI_T2_WD.h5', polar = True, n_layer_cell = [1, 4], cell_radius = 5, n_neighbors = 1000, dis_cutoff = 15, seed = seed_num)
    
    assert adata_query.obs['Recon_scST'].sum()==50305


