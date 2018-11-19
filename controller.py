import numpy as np


def create_sample_population(mu, sigma, amt):
    return np.random.normal(mu, sigma, amt)


def create_histogram(sample, bin_size):
    return np.histogram(sample, bin_size)


def calc_bin_centers(bins):
    return (bins[:-1] + bins[1:]) / 2  # nice


def calc_bin_width(bins):
    return bins[1] - bins[0]


def concatenate_data_sets(data_set_1, data_set_2):
    return np.concatenate([data_set_1, data_set_2])


def calc_probabilities(hist_data, width):
    total_sum = np.sum(hist_data)
    return hist_data[0:].astype(float) / (total_sum * width)


def calc_posterior_non_inf(p_data_1, p_data_2):
    return p_data_1[0:] / (p_data_1[0:] + p_data_2[0:])


def calc_posterior(p_data_1, p_data_2, data_set_1, data_set_2):
    length_1 = len(data_set_1)
    length_2 = len(data_set_2)
    return (p_data_1[0:] * length_1) / (p_data_1[0:] * length_1 + p_data_2[0:] * length_2)


def create_ground_truth_vector(dataset1, dataset2):
    return concatenate_data_sets(np.zeros(len(dataset1)), np.ones(len(dataset2)))


def identify_bin(bin_width, bin_centers, data_point):
    i = 0
    while i < len(bin_centers):
        if data_point <= bin_centers[i] + (bin_width / 2):
            break
        i += 1

    if i == len(bin_centers):
         i = i - 1

    return i


def count_hits(acc_vec, val):
    count = 0
    for value in acc_vec:
        if value == val:
            count += 1
    return count


def train_model(posteriors, total_dataset, bin_width, bin_centers, gt_vec):
    decision_vector = [None]*len(total_dataset)
    accuracy_vector = [None]*len(total_dataset)
    for i in range(0, len(total_dataset) - 1):
        bin_index = identify_bin(bin_width, bin_centers, total_dataset[i])
        if posteriors[bin_index] >= 0.5:  # ToDo: thresholdvalue must be trained
            decision_vector[i] = 1
        else:
            decision_vector[i] = 0

        if decision_vector[i] == gt_vec[i]:
            accuracy_vector[i] = 1
        else:
            accuracy_vector[i] = 0

    hits = count_hits(accuracy_vector, 1.0)
    accuracy = float(hits) / len(accuracy_vector)

    return accuracy
