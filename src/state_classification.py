"""
This script performs the activity state classification procedure as described in the paper.
As it is currently set up, it performs a two-step classification process, reclassifying
the higher energy data if the initial classiciation produces two clusters

@author: Andrew Sonta
"""

import numpy as np
import pandas as pd
import copy
from sklearn import mixture
from scipy import stats
import sys
from bisect import bisect
import __init__

time_step = __init__.TIME_STEP
num_timesteps = None
num_days = None
num_occs = None

num_components = __init__.PRIMARY_COMPONENT_UPPER_BOUND
num_components2 = __init__.SECONDARY_COMPONENT_UPPER_BOUND

# Import Clean Data
def parse():
    script = sys.argv[0]
    file_in = sys.argv[1]
    file_out = sys.argv[2]
    return file_in, file_out

def solve(m1, m2, std1, std2, verbose=False):
    m1 = float(m1)
    m2 = float(m2)
    std1 = float(std1)
    std2 = float(std2)
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    to_ret = np.roots([a,b,c])
    for item in to_ret:
        if ((m1 < item and item < m2) or (m1 > item and item > m2)): return item
    if verbose: print('Cutoff assumed to be midpoint')
    return (to_ret[0] + to_ret[1]) / 2


def labels_from_gmm(gmm, fit_data, verbose=False):
    '''
    This helper function uses data to fit a VB-GMM object and classify the data
    according to the results of the fit. It reassigns labels so that the classifications
    are in increasing order of associated data values

    :param - gmm: Variational Bayesian Gaussian Mixture Model
    :param - fit_data: time-series sensor data

    return type: list
    return: sorted_labels
    '''
    labels = gmm.fit_predict(fit_data)
    used_labels = list(set(labels))
    used_means = sorted([(gmm.means_[x], np.sqrt(gmm.covariances_[x])) for x in used_labels])
    cutoffs = []
    if len(used_means) > 1:
        for c in range(len(used_means)-1):
            cutoff = solve(used_means[c][0], used_means[c+1][0], used_means[c][1], used_means[c+1][1])
            cutoffs.append(cutoff)
    cutoffs.append(float(used_means[len(used_means)-1][0]))

    sorted_labels = []
    for l in range(len(labels)):
        sorted_labels.append(bisect(cutoffs, fit_data[l])+1)
    # for label in labels:
        # sorted_labels.append(used_means.index(gmm.means_[label])+1)
    return sorted_labels

def classification(clean_data, verbose=False):
    '''
    This function performs the classification process

    :param - clean_data: pandas dataframe containing time series plug data

    return type: int
    return: primary_components
    '''
    global num_timesteps
    global num_days
    global num_occs

    clean_data_mat = clean_data.values

    # Create dictionary
    # Key = day
    # Value = data matrix
    arrays_dict = {}
    def day_matrix(mat):
        mat = np.resize(np.transpose(mat),(len(mat)//(num_timesteps//num_days),num_timesteps//num_days))
        return mat
    for i in range(0, len(clean_data_mat[1,:])):
        arrays_dict['x{0}'.format(i)] = day_matrix(clean_data_mat[:,i])

    num_occs = clean_data_mat.shape[1]

    Y_full = np.ndarray(shape=(num_timesteps, num_occs))
    n_components_range = range(1, num_components+1)

    # Loop through each occupant
    all_k = np.ndarray(shape=(num_days, num_occs))
    for i in range(num_occs):
        k = []
        iter_array = np.transpose(arrays_dict['x{0}'.format(i)])
        # Loop through days
        for j in range(num_days):
            fit_data = iter_array[:, j].reshape(-1,1)

            # Create VB-GMM model with up to 6 components
            gmm = mixture.BayesianGaussianMixture(n_components = num_components,
                covariance_type = 'full',
                weight_concentration_prior = 1
                )
            if np.ptp(fit_data) < __init__.VARIATION_THRESHOLD:
                sorted_labels = [1 for x in fit_data]
            elif i in __init__.HIGH_VARIATION_OCCS and np.ptp(fit_data) < __init__.HIGH_VARIATION_THRESHOLD:
                sorted_labels = [1 for x in fit_data]
            else:
                sorted_labels = labels_from_gmm(gmm, fit_data)
            sorted_labels = np.array(sorted_labels)

            # Complete secondary classification process if we find 2 clusters
            if max(sorted_labels) == 2:
                labels_with_index = np.ndarray(shape = (96, 2))
                labels_with_index[:,0] = range(num_timesteps//num_days)
                labels_with_index[:,1] = sorted_labels
                # Make copy without zeros
                labels_with_index_nozeros = labels_with_index[labels_with_index[:,1] != 1].astype(int)
                data_labeled_two = fit_data[labels_with_index_nozeros[:, 0]]
                if len(data_labeled_two) > num_components2:
                    gmm2 = mixture.BayesianGaussianMixture(n_components = num_components2,
                        covariance_type = 'full',
                        weight_concentration_prior = 1)

                    sorted_labels2 = labels_from_gmm(gmm2, data_labeled_two)

                    # labels2 = gmm2.fit_predict(data_labeled_two)
                    # used_labels2 = list(set(labels2))
                    # used_means2 = sorted([(gmm2.means_[x], np.sqrt(gmm2.covariances_[x])) for x in used_labels2])
                    # cutoffs2 = []
                    # if len(used_means2) > 1:
                    #     for c2 in range(len(used_means2)-1):
                    #         cutoff2 = solve(used_means2[c2][0], used_means2[c2+1][0], used_means2[c2][1], used_means2[c2+1][1])
                    #         cutoffs2.append(cutoff2)
                    # cutoffs2.append(float(used_means2[len(used_means2)-1][0]))

                    # sorted_labels2 = []
                    # # for label2 in labels2:
                    # #     sorted_labels2.append(used_means2.index(gmm2.means_[label2])+2)
                    # for l2 in range(len(labels2)):
                    #     sorted_labels2.append(bisect(cutoffs2, fit_data[labels_with_index_nozeros[:,l2]])+2)
                    sorted_labels2 = np.array(sorted_labels2)
                    sorted_labels2[:] += 1
                    sorted_labels[labels_with_index_nozeros[:,0]] = sorted_labels2

            k.append(max(sorted_labels))

            t = num_timesteps//num_days
            Y_full[t*j:t*(j+1), i] = sorted_labels

        all_k[:, i] = k
    if verbose: print('Clusters (columns are occupants, rows are days): \n', all_k)
    if verbose: print('Mode:', int(stats.mode(all_k, axis=None).mode))
    return Y_full

def main():
    '''
    Main workflow for state classification process
    '''
    global num_timesteps
    global num_days
    global num_occs

    file_in, file_out = parse()
    clean_data = pd.read_csv(file_in) 
    clean_data.set_index(str(list(clean_data)[0]), inplace=True)
    num_timesteps = len(clean_data.index)
    num_days = int(num_timesteps / (1440/time_step)) # 15-minute intervals per day
    state_series = classification(clean_data)
    classified_data = pd.DataFrame(state_series, columns = list(clean_data.columns.values))
    classified_data.index = clean_data.index
    # Write classified array to file
    classified_data.to_csv(file_out)

if __name__ == '__main__':
    main()
