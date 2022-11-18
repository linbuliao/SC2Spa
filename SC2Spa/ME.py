import numpy as np
import pandas as pd

import math
from scipy.stats import binom
from multiprocessing import Pool


def Count_Prob_EEI(A):
    '''
    Calculate the counts and probabilities of exclusive events

    Parameters
    ---------
    A
        Gene expression matrix (n_cell x n_gene)


    Returns
    -------
    Count_excl
        Count matrix of exclusive events.
        Prob_excl[i, j]: the probability that gene i is expressed
                                          but gene j is not expressed
    Prob_excl
        Probability matrix of exclusive events.
        Count_excl[i, j]: number of occurrence that gene i is expressed
                                                but gene j is not expressed
    '''

    Allcell = A.shape[0]
    Allgene = A.shape[1]

    Count_excl = np.zeros((Allgene, Allgene), dtype=np.int64)
    is_nonzeroMat = (A.values > 0.05)
    is_zeroMat = (A.values <= 0.05)

    for i in range(Allgene):
        Count_excl[i] = np.sum(np.logical_and(is_nonzeroMat[:, [i]],
                                              is_zeroMat), axis=0)
    p_nonzero = np.sum(A.values > 0.05, axis=0) / Allcell
    p_zero = np.sum(A.values <= 0.05, axis=0) / Allcell

    Prob_excl = p_nonzero * p_zero[:, np.newaxis]

    return Count_excl, Prob_excl


def BME_sub(ij:list, Count_excl:np.array, Prob_excl:np.array, A_S:np.array,
            A_E:np.array, genes:list, n_cell:int, p_coef:int):
    '''
    Calculate Balanced Mutual exclusivity and Directed exclusively express index (subprocess)

    Parameters
    ---------
    ij
        indices of gene pairs
    Count_excl
        Count matrix of exclusive events.
        Prob_excl[i, j]: the probability that gene i is expressed
                                          but gene j is not expressed
    Prob_excl
        Probability matrix of exclusive events.
        Count_excl[i, j]: number of occurrence that gene i is expressed
                                                but gene j is not expressed
    A_S
        boolean matrix, True if a gene is suppressed in one cell. Shape is (cell, gene).
    A_E
        boolean matrix, True if a gene is expressed in one cell. Shape is (cell, gene).
    genes
        A list of genes
    n_cell
        number of cells
    p_coef
        penalty factor

    Returns
    -------
    DEEI
        Mutual exclusivity statistics for given gene pairs ij
    '''

    DEEI = pd.DataFrame(np.zeros((len(ij), 7), dtype=np.float64),
                        columns=['Gene1', 'Gene2', 'BME', 'DEEI(1Express_2Silence)',
                                 'P(1E_2S)', 'DEEI(1Silence_2Express)', 'P(1S_2E)'])

    for count, ijt in enumerate(ij):

        i, j = ijt

        # BME
        e1 = (A_S.iloc[:, i] & A_E.iloc[:, j]).sum()
        e2 = (A_S.iloc[:, j] & A_E.iloc[:, i]).sum()
        e12sumsqr = e1 ** 2 + e2 ** 2
        e1e2 = e1 * e2
        penalty = (e12sumsqr + 2 * e1e2) / e12sumsqr / 2
        e = (e1 + e2) / n_cell * (penalty ** p_coef)

        DEEI.loc[count, 'BME'] = e

        # DEEI
        x1 = Count_excl[i][j]
        p1 = Prob_excl[i][j]
        prob1 = binom.sf(x1 - 1, n_cell, p1)

        x2 = Count_excl[j][i]
        p2 = Prob_excl[j][i]
        prob2 = binom.sf(x2 - 1, n_cell, p2)

        DEEI.loc[count, 'Gene1'] = genes[i]
        DEEI.loc[count, 'Gene2'] = genes[j]
        DEEI.loc[count, 'Prob(1E_2S)'] = prob1
        DEEI.loc[count, 'Prob(1S_2E)'] = prob2

        if (prob1 <= 0):
            deei_1e2s = 1e3
        else:
            deei_1e2s = -(math.log10(prob1))
        if (prob2 <= 0):
            deei_1s2e = 1e3
        else:
            deei_1s2e = -(math.log10(prob2))

        DEEI.loc[count, 'DEEI(1Express_2Silence)'] = deei_1e2s
        DEEI.loc[count, 'DEEI(1Silence_2Express)'] = deei_1s2e

    return DEEI


def BME(A, cutoff=0.05, p_coef=1, n_process=16):
    '''
    Calculate Balanced Mutual Exclusivity and Directed exclusively express index

    Parameters
    ---------
    A
        Gene expression matrix (n_cell x n_gene)
    cutoff
        gene expression cutoff that determines if a gene is expressed
    DEEI
        Mutual exclusivity statistics for given gene pairs ij
    n_process
        number of cores to be used

    Returns
    -------
    DEEI
            Mutual exclusivity statistics
    '''

    Count_excl, Prob_excl = Count_Prob_EEI(A)

    A_S = A < cutoff
    A_E = A >= cutoff

    genes = A.columns.tolist()
    n_cell = A.shape[0]
    n_gene = A.shape[1]

    ij = []
    for i in range(0, n_gene):
        for j in range(i + 1, n_gene):
            ij.append([i, j])

    step = int(len(ij) / n_process)
    ijs = [ij[i * step:(i + 1) * step] for i in range(n_process - 1)]
    ijs.append(ij[(n_process - 1) * step:])

    with Pool(n_process) as p:
        DEEI = p.starmap(BME_sub, [(ij, Count_excl, Prob_excl,
                                    A_S, A_E, genes, n_cell, p_coef) for ij in ijs])

    DEEI = pd.concat(DEEI).sort_values('BME', ascending=False).reset_index(drop=True)

    return DEEI
