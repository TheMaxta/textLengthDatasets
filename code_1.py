import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Load data from CSV files
url1 = 'max_data.csv'
url2 = 'issac_data.csv'
max_data = pd.read_csv(url1)
issac_data = pd.read_csv(url2)


def sample_data(data, n, seed=None):
    """
    Samples n observations from the specified column of the provided dataset.
    Returns the entire rows for these observations.

    :param data: pandas DataFrame from which to sample.
    :param n: The number of samples to draw.
    :param seed: The random seed for reproducibility (default is None).
    :return: A pandas DataFrame containing the sampled rows.
    """
    # Setting the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Sample n observations from the data
    sampled_data = data.sample(n=n)
    return sampled_data

seed = 10
#Create sampled data sets
max_data = sample_data(max_data, 30, seed)
issac_data = sample_data(issac_data, 30, seed)

print("Displaying top five text lengths in dataset.")
max_data.head()


def boxplot(series, name='student'):
    """
    Generates a summary and a box plot from a given pandas Series.

    :param series: pandas Series containing the data.
    :param name: A name for the series (default is 'student').
    """

    # Box plot
    sns.boxplot(x=series)
    plt.title(f"{name}'s Box Plot")
    plt.show()

def five_number_summary(series, name='student'):
    """
    Generates a summary and a box plot from a given pandas Series.

    :param series: pandas Series containing the data.
    :param name: A name for the series (default is 'student').
    """

    # Five number summary
    summary = series.describe()

    # Display summary
    print(f"{name}'s Five Number Summary:\n", summary)

def relative_frequency_histogram(series, name='student', bins=20, xlabel='Value', ylabel='Relative Frequency'):
    # Calculate bin edges
    bin_edges = np.histogram_bin_edges(series, bins=bins)

    # Calculate relative frequency based on these bins
    counts, _ = np.histogram(series, bins=bin_edges)
    relative_freq = counts / counts.sum()

    # Plotting the histogram
    plt.bar(bin_edges[:-1], relative_freq, width=np.diff(bin_edges), edgecolor="black")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(f"{name}'s Relative Frequency Histogram")
    plt.show()

def relative_frequency_table(series, name='Text Length', bins=20, display_table=True):
    # Calculate bin edges (same as used in histogram)
    bin_edges = np.histogram_bin_edges(series, bins=bins)

    # Group the series data into these bins
    binned_series = pd.cut(series, bins=bin_edges, include_lowest=True)

    # Count the frequency of each bin
    frequency = binned_series.value_counts()

    # Calculate relative frequency
    relative_frequency = frequency / frequency.sum()

    # Creating the relative frequency table
    freq_table = pd.DataFrame({f"{name}": frequency.index.astype(str), "Frequency": frequency.values, "Relative Frequency": relative_frequency.values})

    # Display the table if requested
    if display_table:
        print(freq_table.to_string(index=False))

    return freq_table

def confidence_interval(series, confidence=0.95):
    """
    Generates a Confidence Inteveral from a given Pandas Series.

    :param series: pandas Series containing the data.
    :param confidence: represents (1 - significance level), or percentage of confidence
    :return: A pandas DataFrame representing the confidence interval.
    """
    n = len(series)          # Length
    mean = np.mean(series)   # Mean
    sem = stats.sem(series)  # Standard Error
    interval = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean - interval, mean + interval

def compare_box_plot(series1, series2, label1='Series 1', label2='Series 2'):
    """
    Generates a Comparative Box Plot from a given Pandas Series.

    :param series1: pandas Series containing the first dataset for comparison.
    :param series2: pandas Series containing the second dataset for comparison.
    :param label1: label for first dataset
    :param label2: label for second dataset
    """
    # Combine the series into a DataFrame with an identifier
    data1 = pd.DataFrame({label1: series1})
    data2 = pd.DataFrame({label2: series2})
    combined_data = pd.concat([data1, data2], axis=1)

    # Melting the DataFrame for suitable format for seaborn boxplot
    melted_data = combined_data.melt(var_name='Datasets', value_name='Length of Messages')

    # Plotting
    sns.boxplot(x='Datasets', y='Length of Messages', data=melted_data)
    plt.title('Comparative Box Plot')
    plt.show()


def calculate_std_dev_percentages(series):
    """
    Calculates the percentages of observations within 1, 2, and 3 standard deviations of the mean.

    :param series: pandas Series containing the data.
    :return: A dictionary with the percentages for 1, 2, and 3 standard deviations.
    """
    # Ensure the input is a pandas Series
    if not isinstance(series, pd.Series):
        raise ValueError("The input must be a pandas Series.")

    mean = series.mean()
    std_dev = series.std()

    # Define the boundaries for 1, 2, and 3 standard deviations
    boundaries = {
        '1_std_dev': (mean - std_dev, mean + std_dev),
        '2_std_dev': (mean - 2 * std_dev, mean + 2 * std_dev),
        '3_std_dev': (mean - 3 * std_dev, mean + 3 * std_dev)
    }

    percentages = {}
    for key, (lower_bound, upper_bound) in boundaries.items():
        count_within_range = series[(series >= lower_bound) & (series <= upper_bound)].count()
        percentage = (count_within_range / len(series)) * 100
        percentages[key] = percentage

    return percentages
