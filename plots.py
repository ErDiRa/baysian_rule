import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_histogram(hist, bins):
    plt.bar(bins[:-1], hist)
    plt.title("Histogram")
    plt.ylabel("Freq")
    plt.show()


def plot_pdf(total_bin_center, prob_1, prob_2, label_1, label_2):
    plt.step(total_bin_center, prob_1, color="red", label=label_1)
    plt.step(total_bin_center, prob_2, color="blue", label=label_2)
    patch_1 = mpatches.Patch(color="red", label=label_1)
    patch_2 = mpatches.Patch(color="blue", label=label_2)
    plt.title("Probability Density Functions (PDF)")
    plt.xlabel("bin center")
    plt.ylabel("Probability")
    plt.legend(handles=[patch_1, patch_2])
    plt.show()


def plot_posteriors(x_data_1, x_data_2, y_data,  label_1, label_2):
    plt.plot(x_data_1, y_data, color="red", label=label_1)
    plt.plot(x_data_2, y_data, color="blue", label=label_2)
    patch_1 = mpatches.Patch(color="red", label=label_1)
    patch_2 = mpatches.Patch(color="blue", label=label_2)
    plt.title("Posterior")
    plt.xlabel("bin center")
    plt.ylabel("Probability")
    plt.legend(handles=[patch_1, patch_2])
    plt.show()


def plot_posteriors_non_inf(x_data_1, x_data_2, y_data, label_1, label_2):
    plt.plot(x_data_1, y_data, color="red", label=label_1)
    plt.plot(x_data_2, y_data, color="blue", label=label_2)
    patch_1 = mpatches.Patch(color="red", label=label_1)
    patch_2 = mpatches.Patch(color="blue", label=label_2)
    plt.title("Non-informative Posterior")
    plt.xlabel("bin center")
    plt.ylabel("Probability")
    plt.legend(handles=[patch_1, patch_2])
    plt.show()
