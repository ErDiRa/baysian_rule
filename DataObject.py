import controller as clc


class DataObject:
    bin_size = 10

    def __init__(self, dataset):
        self.dataset = dataset
        self.histogram, self.bins = clc.create_histogram(dataset, self.bin_size)
        self.bin_center = clc.calc_bin_centers(self.bins)
        self.bin_width = clc.calc_bin_width(self.bins)
        self.probabilities = clc.calc_probabilities(self.histogram, self.bin_width)

    def get_data_set(self):
        return self.dataset

    def get_frequencies(self):
        return self.histogram

    def get_bins(self):
        return self.bins

    def get_bin_center(self):
        return self.bin_center

    def get_bin_width(self):
        return self.bin_width

    def get_probabilities(self):
        return self.probabilities
