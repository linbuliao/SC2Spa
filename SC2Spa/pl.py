import os

import anndata
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import seaborn as sns

cmap = sns.cubehelix_palette(n_colors = 32,start = 2, rot=1.5, as_cmap = True)
#cmap = 'viridis'

def VisCustomize1(adata:anndata.AnnData, coords_name: str, preds_name: str,
                  doub, root, out, sta, end):
    '''
    Customized function for the paper

    Parameters
    ---------
    adata
    Coordinate Information should be stored in `adata.obsm[coords_name]`
    Predicted Coordinate Information should be stored in `adata.obsm[preds_name]`

    Returns
    -------
    None
    '''
    CTs = adata.obs['MCT'].value_counts()
    is_doub = CTs.index.str.contains('_')

    if(doub):
        SelectedCTs = CTs[is_doub].index.tolist()[sta:end]
    else:
        SelectedCTs = CTs[~is_doub].index.tolist()[sta:end]
    t = adata
    Select1 = t.obs['MCT'].apply(lambda x: x in SelectedCTs)
    Select = Select1
    t = t[Select]

    color_dic = {}
    for i, ct in enumerate(t.obs['MCT'].unique()):
        color_dic[ct] = i

    color = list(map(lambda x: color_dic[x], t.obs['MCT']))

    fig, axes = plt.subplots(1, 2, figsize=(13,5))

    axes[0].scatter(adata.obsm[coords_name][:, 0],
                    adata.obsm[coords_name][:, 1],
                    s = 3, c = color,
                    cmap = plt.get_cmap('Paired'))
    plt.grid(b=None)

    scatter = axes[1].scatter(adata.obsm[preds_name][:, 0],
                              adata.obsm[preds_name][:, 1],
                              s = 3, c = color,
                              cmap = plt.get_cmap('Paired'))
    plt.grid(b=None)
    #plt.axis('off')

    axes[0].title.set_text('Train_raw')
    axes[1].title.set_text('Train_pred')

    #axes[0].set_xlim([650, 5750])
    #axes[0].set_ylim([650, 5750])
    #axes[1].set_xlim([650, 5750])
    #axes[1].set_ylim([650, 5750])

    plt.legend(handles=scatter.legend_elements()[0],
               labels = list(t.obs['MCT'].unique()),
               bbox_to_anchor=(1, 0.9))
    if(doub):
        out = 'doub_' + str(out)
    plt.savefig(root + 'train_overall_' + str(out) + '.pdf', bbox_inches='tight')
    plt.show()


def VisCustomize2(adata: anndata.AnnData, coords_name,
                  preds, ct_name, root, out, sta, end):
    '''
    Customized function for the paper

    Parameters
    ---------
    adata
    Coordinate Information should be stored in `adata.obsm[coords_name]`
    Predicted Coordinate Information should be stored in `adata.obsm[preds_name]`

    Returns
    -------
    None
    '''
    CTs = adata.obs[ct_name].value_counts()

    SelectedCTs = CTs.index.tolist()[sta:end]

    t = adata
    Select1 = t.obs[ct_name].apply(lambda x: x in SelectedCTs)
    Select = Select1
    t = t[Select]
    preds = preds[Select]

    color_dic = {}
    for i, ct in enumerate(t.obs[ct_name].unique()):
        color_dic[ct] = i

    color = list(map(lambda x: color_dic[x], t.obs[ct_name]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(t.obsm[coords_name][:, 0],
                    t.obsm[coords_name][:, 1],
                    s=3, c=color,
                    cmap=plt.get_cmap('Paired'))

    scatter = axes[1].scatter(preds[:, 0], preds[:, 1],
                              s=3, c=color,
                              cmap=plt.get_cmap('Paired'))

    # plt.axis('off')

    axes[0].title.set_text('raw')
    axes[1].title.set_text('pred')

    plt.legend(handles=scatter.legend_elements()[0],
               labels=list(t.obs[ct_name].unique()),
               bbox_to_anchor=(1, 0.9))

    if not os.path.exists(root):
        os.makedirs(root)
    plt.savefig(root + 'Compare_' + str(out) + '.pdf', bbox_inches='tight')
    plt.show()

def DrawSVG(adata:anndata.AnnData, GeneList:pd.DataFrame, target_field: str, coords_name='spatial',
            n_genes = 9, s = 2, lim = False, alpha = 0.8, xlim = [650, 5750], ylim = [650, 5750],
            cmap = cmap, FM = True, scST = False, root = 'figures/SVG/', fontsize=40, CT_name = 'MCT',
            CT = '', Sparse = True, Bottom = False, output = None):

    '''
    Show the expression of location predictive genes (or spatially variable genes)
     in all cells or one cell type

    The integrated figure will be saved as `root + output + '.png' if output!=None`
    The sub-figures will be saved as `root + Genes[i] + '_' + CT + '.png'`

    Parameters
    ---------
    adata
        anndata file.
        Cell type information should be stored in `adata.obs[CT_name]` if `CT!=''`
        Fine mapping result should be stored in `adata.obs['FM']` if `FM==True`
        Coordinate Information should be stored in `adata.obsm[coords_name]`
        Reconstruction info should be stored in adata.obs['Recon_scST']
    CT
        The cell type name to be drawn. Default ''.
    GeneList
        A DataFrame that stores spatial variance information in `target_field` column
    target_field
        The column that stores spatial variance information in `GeneList`
    n_genes
        number of genes to be drawn
    alpha
        Transparency score of beads. 1 is completely opaque. 0 is completely transparent.
    s
        size of beads in the figure
    FM
        Draw finely mapped beads of cells if True
    scST
        Draw reconstructed single-cell ST data if True
    cmap
        colormap. See matplotlib
    fontsize
        font size of title
    Sparse
        True if the gene expression matrix is saved in sparse format
    Bottom
        Prioritize in ascending order if True

    Returns
    -------
    None
    '''

    if (scST):
        adata = adata[adata.obs['Recon_scST']]
    elif(FM):
        adata = adata[adata.obs['FM']]
    if(CT != ''):
        adata = adata[adata.obs[CT_name] == CT]
        target_field = target_field + '_' + CT
        if(output !=None):
            output = output + '_' + CT
    
    GeneList = GeneList.sort_values(target_field, ascending=Bottom)
    Genes = GeneList.index.tolist()

    n_row = int(np.power(n_genes, 0.5))
    n_col = int(np.ceil(np.power(n_genes, 0.5)))

    fig, axes = plt.subplots(n_row, n_col, figsize = (8*n_col, 8*n_row))
    if(lim):
        plt.xlim(xlim)
        plt.ylim(ylim)

    for i in range(n_genes):

        if(Sparse):
            axes[i // n_col][i % n_col].scatter(adata.obsm[coords_name][:, 0],
                                                adata.obsm[coords_name][:, 1],
                                                alpha = alpha, s = s,
                                                c = adata[:, Genes[i]].X.toarray().flatten(),
                                                cmap = cmap)
        else:
            axes[i // n_col][i % n_col].scatter(adata.obsm[coords_name][:, 0],
                                                adata.obsm[coords_name][:, 1],
                                                alpha=alpha, s=s,
                                                c=adata[:, Genes[i]].X.flatten(),
                                                cmap=cmap)

        axes[i//n_col][i%n_col].set_title(Genes[i] + ':' + "%.3f"%GeneList['imp_sumup_norm'][i],
                                          fontsize=fontsize)
        axes[i//n_col][i%n_col].get_xaxis().set_visible(False)
        axes[i//n_col][i%n_col].get_yaxis().set_visible(False)
    
    plt.tight_layout()
    if(output!=None):
        if not os.path.exists(root):
            os.makedirs(root)
        plt.savefig(root + output + '.png', )
    plt.show()

    for i in range(n_genes):
        plt.figure(figsize=(10, 10))
        if(Sparse):
            plt.scatter(adata.obsm[coords_name][:, 0],
                        adata.obsm[coords_name][:, 1],
                        marker='o', s=s, alpha=alpha,
                        c=adata[:, Genes[i]].X.toarray().flatten(),
                        cmap=cmap)
        else:
            plt.scatter(adata.obsm[coords_name][:, 0],
                        adata.obsm[coords_name][:, 1],
                        marker='o', s=s, alpha=alpha,
                        c=adata[:, Genes[i]].X.flatten(),
                        cmap=cmap)

        plt.grid(False)
        plt.axis('off')

        if (output != None):
            if(CT != ''):
                plt.savefig(root + Genes[i] + '_' + CT + '.png', )
            else:
                plt.savefig(root + Genes[i] + '.png', )
        plt.show()


def DrawGenes2(adata:anndata.AnnData, gene:str, coords_name='spatial', lim = False,
               xlim = [650, 5750], ylim = [650, 5750], figsize = (10,10), alpha = 0.7,
               cmap = cmap, FM = True, scST = False, CTL = None, c_name = 'simp_name',
               root = 'transfer2/FM_Valid1/', s = 2, Sparse = True, title = False, save = None):

    '''
    Show the expression of a gene in cells or ST beads

    The figure will be saved as `root + save + '_' + gene + '.png'`

    Parameters
    ---------
    adata
        anndata file.
        Cell type information should be stored in `adata.obs[c_name] if CTL!=None`
        Fine mapping result should be stored in `adata.obs['FM']` if `FM==True`
        Coordinate Information should be stored in `adata.obsm[coords_name]`
        Reconstruction info should be stored in adata.obs['Recon_scST']
    gene
        The gene to be drawn
    CTL
        A list of the names of cell types to be drawn. Default ''.
    s
        size of beads in the figure
    cmap
        colormap. See matplotlib
    FM
        Draw finely mapped beads of cells if True
    scST
        Draw reconstructed single-cell ST data if True
    Sparse
        True if the gene expression matrix is saved in sparse format
    alpha
        control the transparency of the scatters. 1 is transparent.
        0 is opaque.
    figsize
        figure size

    Returns
    -------
    None

    '''
    if(scST):
        adata = adata[adata.obs['Recon_scST']]
    elif(FM):
        adata = adata[adata.obs['FM']]
    if(CTL != None):
        if(c_name != 'SSV2'):
            select = adata.obs[c_name].apply(lambda x: x in CTL)
            adata = adata[select.tolist()]
        else:
            select1 = adata.obs['spot_class'].apply(lambda x: x in ['doublet_certain', 'singlet'])
            select2 = adata.obs['MCT'].apply(lambda x: x in CTL)
            select = select1 & select2
            adata = adata[select.tolist()]

    if(figsize != None):
        plt.figure(figsize=figsize)
    if(lim):
        plt.xlim(xlim)
        plt.ylim(ylim)

    if(Sparse):
        plt.scatter(adata.obsm[coords_name][:, 0], adata.obsm[coords_name][:, 1],\
                    marker='o', s = s, alpha = alpha,\
                    c = adata[:, gene].X.toarray().flatten(),\
                    cmap = cmap)
    else:
        plt.scatter(adata.obsm[coords_name][:, 0], adata.obsm[coords_name][:, 1], \
                    marker='o', s=s, alpha=alpha, \
                    c=adata[:, gene].X.flatten(), \
                    cmap=cmap)

    plt.grid(False)
    plt.axis('off')
    if(title):
        plt.title(gene, fontsize = 12)
    
    if(save!= None):
        if not os.path.exists(root):
            os.makedirs(root)
        
        plt.savefig(root + save + '_' + gene + '.png',
                    dpi = 128, bbox_inches='tight')

    plt.show()


def DrawCT1(adata:anndata.AnnData, CT:str, ax = None, coords_name='spatial', s = 2, FM = True,
            scST = False, c_name='leiden', root='transfer2/FM_Valid2/', figsize = (10, 10), save=None):

    '''
    Show the spatial locations of a type of cells or beads

    The figure will be saved as `root + save + '_' + CT + '.png' if save!=None`

    Parameters
    ---------
    adata
        anndata file.
        Cell type information should be stored in `adata.obs[c_name]
        Fine mapping result should be stored in `adata.obs['FM']` if `FM==True`
        Coordinate Information should be stored in `adata.obsm[coords_name]`
        Reconstruction info should be stored in adata.obs['Recon_scST']
    CT
        The cell type to be drawn. it must be one category in `adata.obs[c_name]`
    ax
        matplotlib ax where the figure is plotted
    coords_name
        Coordinate Information should be stored in `adata.obsm[coords_name]`
    s
        size of beads in the figure
    FM
        Draw finely mapped beads or cells if True.
    scST
        Draw reconstructed single-cell ST data if True.
    figsize
        figure size

    Returns
    -------
    None

    '''
    if(scST):
        adata = adata[adata.obs['Recon_scST']]
    elif(FM):
        adata = adata[adata.obs['FM']]

    if(c_name != 'SSV2'):
        adata_S1 = adata[adata.obs[c_name] == CT]
        adata_S2 = adata[adata.obs[c_name] != CT]
    else:
        select1 = adata.obs['spot_class'].apply(lambda x: x in ['doublet_certain', 'singlet'])
        select2 = (adata.obs['celltype_1'] == CT) | (adata.obs['celltype_2'] == CT)
        adata_S1 = adata[select1 & select2]
        adata_S2 = adata[~(select1 & select2)]

    if((figsize != None)&(ax == None)):
        plt.figure(figsize=figsize)

    if(ax==None):
        plt.scatter(adata_S2.obsm[coords_name][:, 0], adata_S2.obsm[coords_name][:, 1],
                    marker='o', s=s, alpha=0.8, c='lightsteelblue')
        plt.scatter(adata_S1.obsm[coords_name][:, 0], adata_S1.obsm[coords_name][:, 1],
                    marker='o', s=s, alpha=0.8, c='midnightblue')

        plt.grid(False)
        plt.axis('off')
        plt.title(str(CT).replace('?', ''), fontsize=48)

        if ((save != None)):

            if not os.path.exists(root):
                os.makedirs(root)
            CT = str(CT).replace('?', '')
            plt.savefig(root + save + '_' + CT + '.png',
                        dpi=128, bbox_inches='tight')
        plt.show()
    else:
        ax.scatter(adata_S2.obsm[coords_name][:, 0], adata_S2.obsm[coords_name][:, 1],
                    marker='o', s=s, alpha=0.8, c='lightsteelblue')
        ax.scatter(adata_S1.obsm[coords_name][:, 0], adata_S1.obsm[coords_name][:, 1],
                    marker='o', s=s, alpha=0.8, c='midnightblue')
        ax.grid(False)
        ax.axis('off')
        ax.set_title(str(CT).replace('?', ''), fontsize=36)



def DrawCT2(adata, CT:str, coords_name='spatial', title = False, NRD = True,
            cmap=cmap, s=2, root='transfer2/FM_NRD/', save=None):

    '''
    Customized Function

    Show the proportion of one cell type for the beads that are annotated

    The figure will be saved as `root + save + '_' + CT + '.png' if save!=None`

    Parameters
    ---------
    adata
        anndata file.
        Cell type information should be stored in `adata.obs[c_name]
        Fine mapping result should be stored in `adata.obs['FM']` if `FM==True`
        Coordinate Information should be stored in `adata.obsm[coords_name]`
    CT
        The cell type to be drawn. it must be one category in `adata.obs[c_name]`
    s
        size of beads in the figure
    NRD
        True if the NRD (Normalized Reciprocal Distance) result is used
    cmap
        colormap. See matplotlib

    Returns
    -------
    None

    '''
    if(NRD):
        adata = adata[adata.obs['CT_NRD']]

    plt.figure(figsize=(10, 10))

    plt.scatter(adata.obsm[coords_name][:, 0], adata.obsm[coords_name][:, 1], \
                marker='o', s=s, alpha=0.7, \
                c=adata.obs[CT], \
                cmap=cmap)

    plt.grid(False)
    plt.axis('off')
    if(title):
        plt.title(CT, fontsize=48)

    if (save != None):

        if not os.path.exists(root):
            os.makedirs(root)

        plt.savefig(root + save + '_' + CT + '.png',
                    dpi=128, bbox_inches='tight')

    plt.show()


def DrawCT3(adata: anndata.AnnData, CT_list: list, coords_name='spatial_mapping', s=3,
            FM=True, scST=False, c_name='cell_type_med_resolution', legend=False,
            root='Transfer1/FM_Valid3/', figsize=(10, 10), save=None):
    '''
    Display original locations or predicted locations of beads/cells of selected types.
    Beads/cells are colored according to anatomy/cell type.
    A figure will be saved to `root+save+'.png'`
    The legend of the figure will be saved to `root + save + '_legend.png'`

    Parameters
    ---------
    adata
        anndata file.
        Cell type information should be stored in `adata.obs[c_name]
        Fine mapping result should be stored in `adata.obs['FM']` if `FM==True`
        Coordinate Information should be stored in `adata.obsm[coords_name]`
        Reconstruction info should be stored in adata.obs['Recon_scST']
    CT_list
        A list that contains the cell types to be shown. The cell types must
        be in `adata.obs[c_name]`
    c_name
        Name of cell type column in adata.obs
    s
        size of beads in the figure
    legend
        Draw legend in the figure if True

    Returns
    -------
    None

    '''

    if (scST):
        adata = adata[adata.obs['Recon_scST']]
    elif (FM):
        adata = adata[adata.obs['FM']]

    adata_t = adata[adata.obs[c_name].apply(lambda x: x in CT_list)]

    color_dic = {}
    cts = sorted(adata_t.obs[c_name].unique())
    for i, ct in enumerate(cts):
        color_dic[ct] = i

    color = list(map(lambda x: color_dic[x], adata_t.obs[c_name]))

    if (figsize != None):
        plt.figure(figsize=figsize)

    coord = adata_t.obsm[coords_name]

    scatter = plt.scatter(coord[:, 0], \
                          coord[:, 1], \
                          s=s, \
                          c=color, \
                          cmap=plt.get_cmap('Paired'))
    plt.grid(False)
    plt.axis('off')

    if (legend):
        plt.legend(handles=scatter.legend_elements()[0],
                   labels=cts)

    if (save != None):

        if not os.path.exists(root):
            os.makedirs(root)
        plt.savefig(root + save + '.png',
                    dpi=128, bbox_inches='tight')
    plt.show()

    ##Extract handles
    handles = scatter.legend_elements()[0]

    plt.legend(handles=scatter.legend_elements()[0],
               labels=cts)
    plt.grid(False)
    plt.axis('off')

    if(save != None):
        plt.savefig(root + save + '_legend.png', bbox_inches='tight')

    plt.show()


def Superimpose(adata:anndata.AnnData, coords_name='spatial', G1='APOE', G2='NRGN',
                s = 2, C1 = 'Reds', C2 = 'Blues', figsize = (10, 10),
                alpha1=0.5, alpha2=0.7, save_root = 'figures/BME/', save=True):
    '''
    Show the spatial gene expressions of two genes and their superimposed images

    The figures will be saved as if `save==True`:
    `save_root + G1 + '_1.png'` (Spatial Gene Expression of Gene1)
    `save_root + G2 + '_1.png'` (Spatial Gene Expression of Gene2)
    `save_root + G1 + '_' + G2 + '.png'` (Draw Gene1 first in the superimposed figure)
    `save_root + G2 + '_' + G1 + '.png'` (Draw Gene2 first in the superimposed figure)

    Parameters
    ---------
    adata
        anndata file.
        Coordinate Information should be stored in `adata.obsm[coords_name]`
    G1
        name of Gene1
    G2
        name of Gene2
    s
        size of beads in the figure. The second gene in the superimposed figures will
        be drawn in the size of s/4
    alpha1
        Transparency score of beads of the figures for only one gene.
         1 is completely opaque. 0 is completely transparent.
    alpha2
        Transparency score of beads of the superimposed figures.
         1 is completely opaque. 0 is completely transparent.
    C1
        color (map) for the Gene1
    C2
        color (map) for the Gene2

    Returns
    -------
    None

    '''
    if(save):
        if not os.path.exists(save_root):
            os.makedirs(save_root)

    # Gene 1
    plt.figure(figsize=figsize)
    plt.grid(False)
    plt.axis('off')

    plt.scatter(adata.obsm[coords_name][:, 0], \
                adata.obsm[coords_name][:, 1], \
                alpha=alpha1, s=s, \
                c=adata[:, G1].X.toarray().flatten(), \
                cmap=C1)

    if (save):
        plt.savefig(save_root + G1 + '_1.png', bbox_inches='tight')
    plt.show()

    # Gene 2
    plt.figure(figsize=figsize)
    plt.xlim([650, 5750])
    plt.ylim([650, 5750])
    plt.grid(False)
    plt.axis('off')

    plt.scatter(adata.obsm[coords_name][:, 0], \
                adata.obsm[coords_name][:, 1], \
                alpha=alpha1, s=s, \
                c=adata[:, G2].X.toarray().flatten(), \
                cmap=C2)

    if (save):
        plt.savefig(save_root + G2 + '_2.png', bbox_inches='tight')
    plt.show()

    # Superimpose1
    plt.figure(figsize=figsize)
    plt.xlim([650, 5750])
    plt.ylim([650, 5750])
    plt.grid(False)
    plt.axis('off')

    plt.scatter(adata.obsm[coords_name][:, 0], \
                adata.obsm[coords_name][:, 1], \
                alpha=alpha2, s=s, \
                c=adata[:, G1].X.toarray().flatten(), \
                cmap=C1)

    plt.scatter(adata.obsm[coords_name][:, 0], \
                adata.obsm[coords_name][:, 1], \
                alpha=alpha2, s=s/4.0, \
                c=adata[:, G2].X.toarray().flatten(), \
                cmap=C2)

    if (save):
        plt.savefig(save_root + G1 + '_' + G2 + '.png', bbox_inches='tight')
    plt.show()

    # Superimpose2
    plt.figure(figsize=figsize)
    plt.xlim([650, 5750])
    plt.ylim([650, 5750])
    plt.grid(False)
    plt.axis('off')

    plt.scatter(adata.obsm[coords_name][:, 0], \
                adata.obsm[coords_name][:, 1], \
                alpha=alpha2, s=s, \
                c=adata[:, G2].X.toarray().flatten(), \
                cmap=C2)

    plt.scatter(adata.obsm[coords_name][:, 0], \
                adata.obsm[coords_name][:, 1], \
                alpha=alpha2, s=s/4.0, \
                c=adata[:, G1].X.toarray().flatten(), \
                cmap=C1)

    if (save):
        plt.savefig(save_root + G2 + '_' + G1 + '.png', bbox_inches='tight')
    plt.show()

def draw_cb(cmap = 'Reds', figsize = (9, 1.5), save = 'colorbar_reds', size = 28):
    '''
    Draw an independent color bar figure

    The color bar will be saved as `save+".png"`

    Parameters
    ---------
    size
        size of all text
    figsize
        figure size
    cmap
        colormap. See matplotlib

    Returns
    -------
    None

    '''
    from matplotlib import rc
    rc('font', **{'family':'Arial','sans-serif':['Arial']})
    plt.rcParams['font.size'] = size

    a = np.array([[0,1]])
    pl.figure(figsize=figsize)
    img = pl.imshow(a, cmap=cmap)
    pl.gca().set_visible(False)
    cax = pl.axes([0, 0, 0.02, 1.6])
    pl.colorbar(orientation="vertical", cax=cax)
    pl.savefig(save + ".png", bbox_inches='tight')