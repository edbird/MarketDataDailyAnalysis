
import numpy
import matplotlib.pyplot as plt


def plot_log_likelihood_distribution(ll_value_obs=None, ll_values=None, filename=None):

    if ll_value_obs is not None:
        count = 0
        for ll_value in ll_values:
            if ll_value > ll_value_obs:
                count += 1
        cl_value = float(count) / float(len(ll_values))

    print(f'Number of values in histogram: {len(ll_values)}')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Bootstrap log-likelihood value')
    ax.set_ylabel('Number of pseudo-experiments')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    (ll_bin_counts, ll_bin_edges, _) = ax.hist(ll_values, bins=100, histtype='step', color='tab:blue')

    if ll_value_obs is not None:
        x_index = numpy.digitize(ll_value_obs, ll_bin_edges)
        y_value = ll_bin_counts[x_index]
        #ax.plot([ll_value_obs, ll_value_obs], [0.0, y_value], 'k-')

        fill_x = ll_bin_edges[x_index:]
        fill_y = ll_bin_counts[x_index:]
        fill_y = numpy.insert(fill_y, 0, fill_y[0])
        ax.fill_between(fill_x, fill_y, step='pre', facecolor='none', edgecolor='tab:blue', hatch='///')

        s = f'CL = {round((1.0 - cl_value) * 100.0, ndigits=2)} %'
        ax.text(ll_value_obs + 45.0, y_value * 0.1, s, color='tab:blue', horizontalalignment='left')

    fig.savefig(filename)

    if ll_value_obs is not None:
        integral_1 = ll_bin_counts[:x_index].sum()
        integral_2 = ll_bin_counts[x_index:x_index+1].sum()
        integral_3 = ll_bin_counts[x_index+1:].sum()
        print(f'integrals: {integral_1}, {integral_2}, {integral_3}')
        print(f'lengths: {len(ll_bin_counts)}, {len(ll_bin_counts[:x_index])}, {len(ll_bin_counts[x_index:x_index+1])}, {len(ll_bin_counts[x_index+1:])}')
        print(f'cl integral: {float(integral_3 + integral_2) / float(integral_1 + integral_2)}')
        print(f'cl exact: {cl_value}')
        
        
def plot_log_likelihood_distribution_2(ll_value_obs=None, ll_values=None, filenames=None):

    if ll_value_obs is not None:
        count = 0
        integral_count = 0
        for ll_value in ll_values:
            if ll_value > ll_value_obs:
                count += 1
            elif ll_value < ll_value_obs:
                integral_count += 1
                
        integral_value = float(integral_count) / float(len(ll_values))
        cl_value = float(count) / float(len(ll_values))
        # is it really the CL value, or is it the wrong way round?
        # either way, treat as an integral

    print(f'Number of values in histogram: {len(ll_values)}')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Bootstrap log-likelihood value')
    ax.set_ylabel('Number of pseudo-experiments')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    (ll_bin_counts, ll_bin_edges, _) = ax.hist(ll_values, bins=100, histtype='step', color='tab:blue')

    if ll_value_obs is not None:
        x_index = numpy.digitize(ll_value_obs, ll_bin_edges)
        y_value = ll_bin_counts[x_index]
        #ax.plot([ll_value_obs, ll_value_obs], [0.0, y_value], 'k-')

        fill_x = ll_bin_edges[x_index:]
        fill_y = ll_bin_counts[x_index:]
        fill_y = numpy.insert(fill_y, 0, fill_y[0])
        
        fill_x_2 = ll_bin_edges[:x_index+1]
        fill_y_2 = ll_bin_counts[:x_index+1]
        #print(len(fill_y_2))
        #fill_y_2 = numpy.append(fill_y_2, fill_y_2[-1])
        #print(len(fill_y_2))
        
        #ax.fill_between(fill_x, fill_y, step='pre', facecolor='none', edgecolor='tab:blue', hatch='///')
        ax.fill_between(fill_x_2, fill_y_2, step='post', facecolor='none', edgecolor='tab:blue', hatch='\\\\\\')

        s = f'CL = {round((1.0 - cl_value) * 100.0, ndigits=2)} %'
        s = f'$\int$ = {round(integral_value * 100.0, ndigits=2)} %'
        #ax.text(ll_value_obs + 45.0, y_value * 0.1, s, color='tab:blue', horizontalalignment='left')
        #ax.plot([ll_value_obs, ll_value_obs], [0.0, y_value], color='tab:red')
        ax.text(ll_value_obs + 5.0, y_value * 0.05, s, color='tab:blue',
                #edgecolor='tab:blue',
                backgroundcolor='white', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='tab:blue', pad=3.0))

    for filename in filenames:
        fig.savefig(filename)

    if ll_value_obs is not None:
        integral_1 = ll_bin_counts[:x_index].sum()
        integral_2 = ll_bin_counts[x_index:x_index+1].sum()
        integral_3 = ll_bin_counts[x_index+1:].sum()
        print(f'integrals: {integral_1}, {integral_2}, {integral_3}')
        print(f'lengths: {len(ll_bin_counts)}, {len(ll_bin_counts[:x_index])}, {len(ll_bin_counts[x_index:x_index+1])}, {len(ll_bin_counts[x_index+1:])}')
        print(f'cl integral: {float(integral_3 + integral_2) / float(integral_1 + integral_2)}')
        print(f'cl exact: {cl_value}')


def plot_log_likelihood_distribution_both(ll_value_obs=None, ll_values=None, ll_values_fail=None, filename=None):

    if ll_value_obs is not None:
        count = 0
        for ll_value in ll_values:
            if ll_value > ll_value_obs:
                count += 1
        cl_value = float(count) / float(len(ll_values))

    print(f'Number of success values in histogram: {len(ll_values)}')
    print(f'Number of fail values in histogram: {len(ll_values_fail)}')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Bootstrap log-likelihood value')
    ax.set_ylabel('Number of pseudo-experiments')

    (ll_bin_counts, ll_bin_edges, _) = ax.hist(ll_values, bins=100, histtype='step', color='tab:blue')
    (ll_bin_counts_fail, ll_bin_edges_fail, _) = ax.hist(ll_values_fail, bins=100, histtype='step', color='tab:orange', range=(ll_bin_edges[0], ll_bin_edges[-1]))

    if ll_value_obs is not None:
        x_index = numpy.digitize(ll_value_obs, ll_bin_edges)
        y_value = ll_bin_counts[x_index]
        #ax.plot([ll_value_obs, ll_value_obs], [0.0, y_value], 'k-')

        fill_x = ll_bin_edges[x_index:]
        fill_y = ll_bin_counts[x_index:]
        fill_y = numpy.insert(fill_y, 0, fill_y[0])
        ax.fill_between(fill_x, fill_y, step='pre', facecolor='none', edgecolor='tab:blue', hatch='///')

        s = f'CL = {round((1.0 - cl_value) * 100.0, ndigits=2)} %'
        ax.text(ll_value_obs - 1.0, y_value * 0.5, s, color='tab:blue', horizontalalignment='right')

    fig.savefig(filename)

    if ll_value_obs is not None:
        integral_1 = ll_bin_counts[:x_index].sum()
        integral_2 = ll_bin_counts[x_index:x_index+1].sum()
        integral_3 = ll_bin_counts[x_index+1:].sum()
        print(f'integrals: {integral_1}, {integral_2}, {integral_3}')
        print(f'lengths: {len(ll_bin_counts)}, {len(ll_bin_counts[:x_index])}, {len(ll_bin_counts[x_index:x_index+1])}, {len(ll_bin_counts[x_index+1:])}')
        print(f'cl integral: {float(integral_3 + integral_2) / float(integral_1 + integral_2)}')
        print(f'cl exact: {cl_value}')