
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