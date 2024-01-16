
import numpy
import matplotlib.pyplot as plt


def plot_log_likelihood_outlier_no_outlier(
    optimized_parameters_1, optimized_parameters_2,
    data_1_log_likelihood, data_2_log_likelihood,
    filenames):
    
    print(f'data_1_log_likelihood={data_1_log_likelihood}')
    print(f'data_2_log_likelihood={data_2_log_likelihood}')

    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    axis.set_xlabel('Bootstrap log-likelihood value')
    axis.set_ylabel('Number of pseudo-experiments')
    axis.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    log_likelihood_values_1 = numpy.array(optimized_parameters_1)[:,2]
    log_likelihood_values_2 = numpy.array(optimized_parameters_2)[:,2]
    
    (ll_bin_counts_2, ll_bin_edges_2, handle_2) = axis.hist(
        log_likelihood_values_2, bins=100, histtype='step', color='tab:orange')
    
    (ll_bin_counts_1, ll_bin_edges_1, handle_1) = axis.hist(
        log_likelihood_values_1, bins=100, histtype='step', color='tab:blue', range=(ll_bin_edges_2[0], ll_bin_edges_2[-1]))
    
    labels = ['Default', 'with Outliers']
    axis.legend(handles=[handle_1[0], handle_2[0]], labels=labels)
    
    x_index_1 = numpy.digitize(data_1_log_likelihood, ll_bin_edges_1)
    x_index_2 = numpy.digitize(data_2_log_likelihood, ll_bin_edges_2)
    
    y_value_1 = ll_bin_counts_1[x_index_1]
    y_value_2 = ll_bin_counts_2[x_index_2]
    
    axis.plot([data_1_log_likelihood, data_1_log_likelihood], [0.0, y_value_1], color='tab:blue')
    axis.plot([data_2_log_likelihood, data_2_log_likelihood], [0.0, y_value_2], color='tab:orange')
    
    for filename in filenames:
        figure.savefig(filename)