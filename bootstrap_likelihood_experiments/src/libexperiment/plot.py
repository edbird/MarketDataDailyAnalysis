
import numpy
import matplotlib.pyplot as plt


def plot_log_likelihood_distribution(optimized_parameters, filenames):
    
    log_likelihood_values = numpy.array(optimized_parameters)[:,2]
    
    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    axis.set_xlabel('Bootstrap log-likelihood value')
    axis.set_ylabel('Number of pseudo-experiments')
    axis.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    (ll_bin_counts, ll_bin_edges, _) = axis.hist(
        log_likelihood_values, bins=100, histtype='step', color='tab:blue')
    
    for filename in filenames:
        figure.savefig(filename)


def plot_log_likelihood(optimized_parameters, data_log_likelihood, filenames):
    
    log_likelihood_values = numpy.array(optimized_parameters)[:,2]
    
    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    axis.set_xlabel('Bootstrap log-likelihood value')
    axis.set_ylabel('Number of pseudo-experiments')
    axis.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    (ll_bin_counts, ll_bin_edges, _) = axis.hist(
        log_likelihood_values, bins=100, histtype='step', color='tab:blue')
    
    print(f'Data Log-Likelihood: {data_log_likelihood}')
    
    for filename in filenames:
        figure.savefig(filename)
    

def plot_log_likelihood_list(optimized_parameters_list, data_log_likelihood, filenames, labels):
    
    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    axis.set_xlabel('Bootstrap log-likelihood value')
    axis.set_ylabel('Number of pseudo-experiments')
    axis.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    handles = []
    
    for optimized_parameters in optimized_parameters_list:
        log_likelihood_values = numpy.array(optimized_parameters)[:,2]
    
        (ll_bin_counts, ll_bin_edges, handle) = axis.hist(
            log_likelihood_values, bins=100, histtype='step')
        
        handles.append(handle[0])
    
    axis.legend(handles=handles, labels=labels, fontsize=6, ncol=3)
    
    print(f'Data Log-Likelihood: {data_log_likelihood}')
    
    for filename in filenames:
        figure.savefig(filename)
    
    