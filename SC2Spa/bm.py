import anndata
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from functools import reduce
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import bootstrap

from IPython.display import clear_output

#from matplotlib import rc
#rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})

import seaborn as sns
cmap = sns.cubehelix_palette(n_colors = 32,start = 2, rot=1.5, as_cmap = True)

#Number of columes for the prediction
#Global variable
n_col = 2


def BVMI(adata1: anndata.AnnData, adata2: anndata.AnnData,
         GL: list, coords_name = 'spatial', n_neighbors=300, max_dis=1700):

    '''
    Calculate bivariate Moran's I of the genes of two ST data.

    The two ST datasets share some or all ST voxels/cells

    Parameters
    ---------
    adata1
        a anndata object.
    adata2
        a anndata object that has the same voxels as or share part of the voxels
        with adata1. The obs_names of the two objects should be consistent to
        extract the overlapped voxels.
    GL: The list of genes to be calculated. The genes should be in var_names of
        adata1 and adata2
    coords_name
        the spatial information should be stored in adata2.obsm[coords_name]
    n_neighbors
        the number of max neighbors for the knn algorithm
    max_dis
        max distance of two neighbors

    Returns
    -------
    MoranI_BV
        a dataframe contains the gene name information and bivariate
         moran's I values
    '''

    MoranI_BV = pd.DataFrame(np.zeros((len(GL), 2)), columns=['gene', 'I_BV'])
    MoranI_BV['gene'] = GL
    MoranI_BV.index = MoranI_BV['gene']

    JCs = sorted(list(set(adata1.obs_names).intersection(set(adata2.obs_names))))
    adata1 = adata1[JCs]
    adata2 = adata2[JCs]

    neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=max_dis)
    neigh.fit(adata2.obsm[coords_name])
    dis, neighbors = neigh.kneighbors(adata2.obsm[coords_name], n_neighbors)

    min_dis = dis[:, 1:].min()
    dis[dis < min_dis] = min_dis

    mask = (dis < max_dis).sum(axis=1)

    for i, gene in enumerate(GL):

        Dev1 = adata1[:, gene].X.toarray() - adata1[:, gene].X.mean()
        Dev2 = adata2[:, gene].X.toarray() - adata2[:, gene].X.mean()

        numerator = 0
        W0 = 0
        n = adata2.shape[0]

        for j in range(n):

            if (mask[j] < 1):
                continue
            t1 = neighbors[j][:mask[j]]
            t2 = 100 / dis[j][:mask[j]]
            W0 += t2.sum()
            length = len(t1)
            t_sum = (Dev1[[t1[0]] * length].flatten() * Dev2[t1].flatten() * t2).sum()
            # Sum up "t_sum" of all genes
            numerator += t_sum

        # Standardization
        MoransI_BV = numerator / ((Dev1 ** 2).sum() ** 0.5) / ((Dev2 ** 2).sum() ** 0.5) / W0 * n

        print(i, gene, ':', MoransI_BV)
        MoranI_BV.loc[gene, 'I_BV'] = MoransI_BV
        clear_output(wait=True)

    return MoranI_BV


def my_pearsonr(Y_pred, Y_true):
    '''
    Function for caculaating the bootstraping confidence interval of
    Pearson Correlation Coefficient of the predicted locations
      and the original locations

    Parameters
    ---------
    Y_pred
        The predicted locations
    Y_true
        The original locations

    Returns
    -------
    Pearson Correlation Coefficient of the predicted locations
      and the original locations
    '''

    return stats.pearsonr(Y_pred, Y_true)[0]


def my_r2(Y_pred, Y_true):
    '''
    Function for caculaating the bootstraping confidence interval of
    Coefficient of Determination between the predicted locations
      and the original locations

    Parameters
    ---------
    Y_pred
        The predicted locations
    Y_true
        The original locations

    Returns
    -------
    Coefficient of Determination between the predicted locations
      and the original locations
    '''

    Y_true = Y_true.reshape((-1, n_col))
    Y_pred = Y_pred.reshape((-1, n_col))

    return r2_score(Y_true, Y_pred, multioutput='variance_weighted')


def my_rmse(Y_pred, Y_true):
    '''
    Function for caculaating the bootstraping confidence interval of
    Root Mean Square Error between the predicted locations
      and the original locations

    Parameters
    ---------
    Y_pred
        The predicted locations
    Y_true
        The original locations

    Returns
    -------
    Root Mean Square Error of the predicted locations
      and the original locations
    '''

    Y_true = Y_true.reshape((-1, n_col))
    Y_pred = Y_pred.reshape((-1, n_col))

    return np.sqrt(np.square(Y_true - Y_pred).sum() / Y_true.shape[0])


def evaluate_CI(Y_pred: np.array, Y_true: np.array, n_resamples=200):
    '''
    Caculaate the Pearson Correlation Coefficient, Root Mean Square Error
     and Coefficient of Determination between the predicted locations
    and the original locations and corresponding confidence intervals

    Parameters
    ---------
    Y_pred
        The predicted locations
    Y_true
        The original locations
    n_resamples
        The number of resamples performed to form the bootstrap distribution of the statistic

    Returns
    -------
    statistics
        A list storing the values of Pearson Correlation Coefficient, Root Mean Square Error
         and Coefficient of Determination.
    CI_BS
        A list containing the confidence interval objects of the scipy.stats.bootstrap function
        of the corresponding statistics.
    '''

    # Calculate confidence interval from bootstrap
    rng = np.random.default_rng()

    CI_pearsonr = bootstrap((Y_true.flatten(), Y_pred.flatten()), my_pearsonr, n_resamples=n_resamples,
                            vectorized=False, paired=True, method='percentile', random_state=rng)
    CI_rmse = bootstrap((Y_true.flatten(), Y_pred.flatten()), my_rmse, paired=True, vectorized=False,
                        n_resamples=n_resamples, method='basic', random_state=rng)
    CI_r2 = bootstrap((Y_true.flatten(), Y_pred.flatten()), my_r2, paired=True, vectorized=False,
                      n_resamples=n_resamples, method='basic', random_state=rng)
    CI_BS = [CI_pearsonr, CI_rmse, CI_r2]

    # Calculate statistics
    # pearsonr = []
    # for i in range(Y_true.shape[1]):
    #    t = stats.pearsonr(Y_true[:, i], Y_pred[:, i])[0]
    #    pearsonr.append(t)
    pearsonr = stats.pearsonr(Y_true.flatten(), Y_pred.flatten())[0]
    rmse = np.sqrt(np.square(Y_true - Y_pred).sum() / Y_true.shape[0])
    r2 = r2_score(Y_true, Y_pred, multioutput='variance_weighted')
    # statistics = [np.array(pearsonr).mean(), rmse, r2]
    statistics = [pearsonr, rmse, r2]

    return statistics, CI_BS


def evaluate(Y_pred: np.array, Y_true: np.array):
    '''
    Caculaate the Pearson Correlation Coefficient, Root Mean Square Error
     and Coefficient of Determination between the predicted locations
      and the original locations

    Parameters
    ---------
    Y_pred
        The predicted locations
    Y_true
        The original locations

    Returns
    -------
    A list that stores the values of Pearson Correlation Coefficient, Root Mean Square Error
     and Coefficient of Determination
    '''

    pearsonr = []
    for i in range(Y_true.shape[1]):
        t = stats.pearsonr(Y_true[:, i], Y_pred[:, i])[0]
        pearsonr.append(t)

    #N = reduce(lambda x, y: x*y, Y_true.shape)

    rmse = np.sqrt(np.square(Y_true - Y_pred).sum() / Y_true.shape[0])
    r2 = r2_score(Y_true, Y_pred, multioutput='variance_weighted')
    out = [np.array(pearsonr).mean(), rmse, r2]

    return out


def obtain_PDs(indices, CP, Y_Norm_all, Y_pred):
    Original_PC1 = Y_Norm_all[CP[:, 0].astype(int)]
    Original_PC2 = Y_Norm_all[CP[:, 1].astype(int)]
    PD_true = ((Original_PC2 - Original_PC1) ** 2).sum(axis=1) ** 0.5

    pred = pd.DataFrame(Y_pred, index=indices)
    pred_PC1 = pred.loc[CP[:, 0].astype(int)].values
    pred_PC2 = pred.loc[CP[:, 1].astype(int)].values
    PD_pred = ((pred_PC2 - pred_PC1) ** 2).sum(axis=1) ** 0.5

    return PD_pred, PD_true


def PD_rmse(PD_pred, PD_true):
    out = ((PD_pred - PD_true) ** 2).mean() ** 0.5

    return out


def eval_PD_rmse(YNorm, test_indices, preds, CPs):
    PDRMSEs = []
    PD_Pearsons = []

    for i in range(len(test_indices)):
        PD_pred, PD_true = obtain_PDs(indices=test_indices[i], CP=CPs[i],
                                      Y_Norm_all=YNorm, Y_pred=preds[i])
        PDRMSE = PD_rmse(PD_pred, PD_true)
        PD_Pearson = stats.pearsonr(PD_pred, PD_true)[0]

        PDRMSEs.append(PDRMSE)
        PD_Pearsons.append(PD_Pearson)

    return PDRMSEs, PD_Pearsons

def eval_PD_CI(PD_true, PD_pred, n_resamples = 200):
    '''
    Caculaate the Pearson Correlation Coefficient, Root Mean Square Error
    of the predicted pairwise distances and the true pairwise distances
     and corresponding confidence intervals

    Parameters
    ---------
    PD_true
        The original normalized distances of bead/cell pairs
    PD_pred
        The predicted normalized distances of bead/cell pairs

    Returns
    -------
    statistics
        A list storing the values of Pearson Correlation Coefficient and Root Mean Square Error
        of the original and predicted pairwise distances
    CI_BS
        A list containing the confidence interval objects of the scipy.stats.bootstrap function
        of the corresponding statistics.
    '''

    rng = np.random.default_rng()

    CI_PD_pearsonr = bootstrap((PD_true, PD_pred), my_pearsonr, n_resamples=n_resamples,
                               vectorized=False, paired=True, method='percentile', random_state=rng)
    CI_PD_rmse = bootstrap((PD_true, PD_pred), PD_rmse, n_resamples=n_resamples,
                           vectorized=False, paired=True, method='percentile', random_state=rng)
    CI_BS = [CI_PD_pearsonr, CI_PD_rmse]

    PDRMSE = PD_rmse(PD_pred, PD_true)
    PD_Pearson = stats.pearsonr(PD_pred, PD_true)[0]
    statistics = [PD_Pearson, PDRMSE]

    return statistics, CI_BS

def eval_cv(preds: list, YNorm: np.array, test_indices: list, CPs: np.array,
            CI_BS = False, n_resamples=200):
    '''
    Caculaate the Pearson Correlation Coefficient, Root Mean Square Error
     and Coefficient of Determination between the predicted locations
      and the original locations for the test beads in Cross-Validation.
    Corresponding confidence intervals will be calculated if `CI_BS` is True.

    Parameters
    ---------
    preds
        The predicted min-max normalized locations for the test beads in cross-validation
    YNorm
        The original min-max normalized locations of all beads
    test_indices
        The indices of test beads in cross-validation
    CPs
        Cell pairs in the format of numpy array
    CI_BS
        Calculate confidence interval using bootstrap if True
    n_resamples
        The number of resamples performed to form the bootstrap distribution of the statistic

    Returns
    -------
    statistics
         A DataFrame that stores the Pearson Correlation Coefficient,
         Root Mean Square Error and Coefficient of Determination of each repetition of Cross-Validation.
    CI_BS
         The confidence intervals for the statistics. This will be output only when `CI_BS` is True.
    '''

    if(CI_BS):
        statistics, CI_BS = evaluate_CI(Y_pred=np.concatenate([preds[j] for j in range(5)]),
                                     Y_true=np.concatenate([YNorm[test_indices[j]] for j in range(5)]),
                                     CI_BS=CI_BS, n_resamples=n_resamples)

        return statistics, CI_BS
    else:
        #Calculate statistics
        performances = []

        for i in range(len(preds)):
            performance = evaluate(Y_pred = preds[i],
                                   Y_true = YNorm[test_indices[i]])
            performances.append(performance)

        statistics = pd.DataFrame(performances,
                                  columns = ['pearsonr', 'RMSE', 'R2'])
        PDRMSEs, PD_Pearsons = eval_PD_rmse(YNorm = YNorm, test_indices = test_indices, preds = preds,
                                 CPs = CPs)
        statistics['PD_RMSE'] = PDRMSEs
        statistics['PD_Pearson'] = PD_Pearsons

        return statistics

    
def Visualize_SSV2(adata:anndata.AnnData, coord:np.array, out_prefix = 'Benchmarking/MH1/',
		           ctname = 'MCT', anchor = (1, 0.9), s = 3,
	               xlim = [650, 5750], ylim = [650, 5750], lim = False,
	               title = 'SC2Spa', legend = False):
    '''
    Display original locations or predicted locations of beads.
    Beads are colored according to cell type.
    Eight most common cell types will be visualized.
    A figure will be saved to `out_prefix+title+'_no_legend.png'`
    The legend of the figure will be saved to `out_prefix + 'legend.png'`

    Parameters
    ---------
    adata
        A AnnData object. Cell type information is stored in adata.obs[ctname].
    coord
        Coordinates of beads
    ctname
        Name of cell type column in adata.obs
    s
        size of beads in the figure
    legend
        Draw legend in the figure if True

    Returns
    -------
    None

    '''

    if not os.path.exists(out_prefix):
        os.makedirs(out_prefix)
    
    CTs = adata.obs[ctname].value_counts()
    if(ctname=='MCT'):
        is_doub = CTs.index.str.contains('_')
        SelectedCTs = CTs[~is_doub].index.tolist()[:8]
    else:
        SelectedCTs = CTs.index.tolist()[:8]

    t = adata
    Select1 = t.obs[ctname].apply(lambda x: x in SelectedCTs).tolist()
    Select = Select1
    t = t[Select]
    coord = coord[Select]

    color_dic = {}
    for i, ct in enumerate(t.obs[ctname].unique()):
        color_dic[ct] = i

    color = list(map(lambda x: color_dic[x], t.obs[ctname]))

    plt.figure(figsize =(7,6))
    scatter = plt.scatter(coord[:, 0],\
                coord[:, 1],\
                s = s,\
                c = color,\
                cmap = plt.get_cmap('Paired'))
    plt.grid(False)
    plt.tick_params(axis='both',
                    which='both',
                    bottom=False,
                    labelbottom=False,
                    left=False,
                    labelleft=False)
    
    if(lim):
        plt.xlim(xlim)
        plt.ylim(ylim)

    ##Extract handles
    handles = scatter.legend_elements()[0]

    if(legend):
        plt.title(title)
        plt.legend(handles=scatter.legend_elements()[0],
                   labels = list(t.obs[ctname].unique()),
                   bbox_to_anchor=anchor)
        plt.savefig(out_prefix + title + '.png', bbox_inches='tight')
    else:
        plt.savefig(out_prefix + title + '_no_legend.png', bbox_inches='tight')

    plt.show()

    ###################
    ####Draw legend####
    ###################
    # Create legend
    plt.legend(handles=scatter.legend_elements()[0],
               labels=list(t.obs[ctname].unique()))
    # Get current axes object and turn off axis
    plt.gca().set_axis_off()
    plt.savefig(out_prefix + 'legend.png', bbox_inches='tight')


def Vis_Euclidean(coord_ref: np.array, coord_pred: np.array, max_value: float,
         out_prefix='BM_figures_2/MH1/', figsize=(8, 6), s=3, xlim=[650, 5750],
         ylim=[650, 5750], cmap=cmap, lim=False, save='SC2Spa'):
    '''
    Show the euclidean distance of the original and predicted locations of beads.
    Beads are colored by the distance.

    Parameters
    ---------
    adata
        A AnnData object. Cell type information is stored in adata.obs[ctname].
    coord_ref
        True coordinates of beads
    coord_pred
        Predicted coordinates of beads
    s
        size of beads in the figure
    save
        save the figure in `out_prefix` + `save` + '.png' if save is not None

    Returns
    -------
    None

    '''

    if not os.path.exists(out_prefix):
        os.makedirs(out_prefix)

    dis = ((coord_pred - coord_ref) ** 2).sum(axis=1) ** 0.5

    plt.figure(figsize=figsize)
    plt.scatter(coord_ref[:, 0], coord_ref[:, 1],
                s=s, c=dis, cmap=cmap, vmax = max_value)
    plt.grid(False)
    plt.tick_params(axis='both',
                    which='both',
                    bottom=False,
                    labelbottom=False,
                    left=False,
                    labelleft=False)
    plt.colorbar()
    if (lim):
        plt.xlim(xlim)
        plt.ylim(ylim)
    if (save != None):
        plt.savefig(out_prefix + save + '.png', bbox_inches='tight')
    plt.show()


def Barplot(BMs:list, BMs_std = None, fontsize = 12, legend = False, save_root = 'Benchmarking/',
	        colors = ['C3', 'C5', 'C4', 'C1', 'C0'], fill = True):
    '''
    Benchmark multiple tools on multiple datasets.
    A figure will be saved to `'Benchmarking/'+column+'_no_legend.png'`
    The legend of the figure will be saved to `'Benchmarking/barplot_legend.png'`

    Parameters
    ---------
    BMs
        Each element is a DataFrame that stores the statistics of multiple tools
         on a dataset. Shape (n_tool, n_statistics).
    BMs_std
        Each element is a DataFrame that stores the standard deviation of multiple tools
         on a dataset. Shape (n_tool, n_statistics). Default None.
    fontsize
        font size of all text
    legend
        Draw legend in the figure if Ture
    colors
        Code of colors for different tools. Available colors see matplotlib.pyplot
    save_root
        Directory to save figures
    fill
        fill the bars with specified colors if True

    Returns
    -------
    None

    '''
    #Valid font size are xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller, None
    ind = np.arange(len(BMs))
    width = 0.15

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    plt.rcParams['font.size'] = fontsize

    for column in BMs[0].columns:
        print(column)
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        for i in range(BMs[0].shape[0]):
            if(BMs_std != None):
                ax.bar(ind + width * i, [float(BM.iloc[i][column]) for BM in BMs],
                       fill=fill, color=colors[i], width=width - 0.04,
                       yerr=[float(BM_std.iloc[i][column]) for BM_std in BMs_std],
                       capsize=5)
            else:
                ax.bar(ind + width * i, [float(BM.iloc[i][column]) for BM in BMs],
                       fill = fill, color = colors[i], width = width-0.04)

        plt.tick_params(axis='x',
                        which='both',
                        bottom=False,
                        labelbottom=False)
        if(legend):
            ax.legend(labels=BMs[0].index.tolist())
            plt.title(column, fontsize = 24)
        if(legend):
            plt.savefig(save_root + '/' + column + '.png', bbox_inches = 'tight')
        else:
            plt.savefig(save_root + '/' + column + '_no_legend.png', bbox_inches = 'tight')
        plt.show()

    ###################
    ####Draw legend####
    ###################
    palette = dict(zip(BMs[0].index.tolist(), colors))
    # Create legend handles manually
    handles = [mpl.patches.Patch(color=palette[x], label = x, fill = fill) for x in palette.keys()]
    # Create legend
    plt.legend(handles=handles)
    # Get current axes object and turn off axis
    plt.gca().set_axis_off()
    plt.savefig(save_root + '/barplot_legend.png', bbox_inches='tight')
