import numpy as np
import controller as ctrl
import plots
from DataObject import DataObject

show_hist = False
show_pdfs = False
show_post_plots = True

threshold = 0.5


def main():

    global show_hist
    global show_pdfs
    global show_post_plots
    global threshold

    x_seabass = ctrl.create_sample_population(30, 3, 400)
    x_salmon = ctrl.create_sample_population(32, 2, 600)

    # Create the dataset objects
    sea_bass = DataObject(x_seabass)
    salmon = DataObject(x_salmon)
    total = DataObject(ctrl.concatenate_data_sets(x_seabass, x_salmon))

    # Check p-sums
    print("----------------------------------------------------------------------------------------------------------")
    print("Sums probabilities")
    sum_sea_bass = np.sum(sea_bass.get_probabilities()[0:]*sea_bass.get_bin_width())
    sum_salmon = np.sum(salmon.get_probabilities()[0:]*salmon.get_bin_width())
    print("Seabass: %f" % sum_sea_bass)
    print("Salmon: %f" % sum_salmon)
    print("----------------------------------------------------------------------------------------------------------")

    # ToDo: calculate decision vector
    # ToDo: calculate Accuracy
    # ToDo: Test samples to trained Model

    print("----------------------------------------------------------------------------------------------------------")
    print("Probabilities")
    p_sea_bass = sea_bass.get_probabilities()
    p_salmon = salmon.get_probabilities()
    print("Seabass: %s" % str(p_sea_bass))
    print("Salmon: %s" % str(p_salmon))
    print("----------------------------------------------------------------------------------------------------------")

    # Posteriors informative
    print("----------------------------------------------------------------------------------------------------------")
    print("Informative posteriors")
    p_sea_bass = sea_bass.get_probabilities()
    p_salmon = salmon.get_probabilities()
    post_sea_bass = ctrl.calc_posterior(p_sea_bass, p_salmon, sea_bass.get_data_set(), salmon.get_data_set())
    post_salmon = ctrl.calc_posterior(p_salmon, p_sea_bass, salmon.get_data_set(), sea_bass.get_data_set())
    print("Seabass: %s" % str(post_sea_bass))
    print("Salmon: %s" % str(post_salmon))
    print("----------------------------------------------------------------------------------------------------------")

    # Non informative posteriors
    print("----------------------------------------------------------------------------------------------------------")
    print("Non-Informative posteriors")
    post_sea_bass_non_inf = ctrl.calc_posterior_non_inf(p_sea_bass, p_salmon)
    post_salmon_non_inf = ctrl.calc_posterior_non_inf(p_salmon, p_sea_bass)
    print("Seabass: %s" % str(post_sea_bass_non_inf))
    print("Salmon: %s" % str(post_salmon_non_inf))
    print("----------------------------------------------------------------------------------------------------------")

    # Create Groundtrough Vector

    ground_truth_vec = ctrl.create_ground_truth_vector(x_salmon, x_seabass)

    # Train Model
    print("----------------------------------------------------------------------------------------------------------")
    print("Training of Model")
    result_training = ctrl.train_model(post_salmon_non_inf, total.get_data_set(), salmon.get_bin_width(),
                                       salmon.get_bin_center(), ground_truth_vec)

    print("Accuracy: %s" % str(result_training))
    print("----------------------------------------------------------------------------------------------------------")

    # Plots
    if show_post_plots:
        bin_center_total = total.get_bin_center()
        plots.plot_posteriors(post_sea_bass, post_salmon, bin_center_total, "Seabass", "Salmon")
        plots.plot_posteriors_non_inf(post_sea_bass_non_inf, post_salmon_non_inf, bin_center_total, "Seabass", "Salmon")

    if show_pdfs:
        plots.plot_pdf(total.get_bin_center(), sea_bass.get_probabilities(), salmon.get_probabilities(),
                       "Seabass", "Salmon")

    if show_hist:
        plots.plot_histogram(sea_bass.get_frequencies(), sea_bass.get_bins())
        plots.plot_histogram(salmon.get_frequencies(), salmon.get_bins())
        plots.plot_histogram(total.get_frequencies(), total.get_bins())


if __name__ == "__main__":
    main()
