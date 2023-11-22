


def plot_bootstrap_data(iteration, bootstrap_sample):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Diff')
    ax.set_ylabel('Probability, AAPL Close Change, Bootstrap Data')
    ax.grid(True, axis='both', alpha=0.3, which='both')

    (bin_contents, bin_edges, _) = \
        ax.hist(bootstrap_sample, label='AAPL', bins=20, density=False)

    fig.savefig(f'bootstrap_data_figure/bootstrap_{iteration}.png')
    plt.close()

