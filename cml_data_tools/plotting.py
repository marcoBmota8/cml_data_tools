"""
Tools for plotting objects
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_phenotypes_to_file(phenotypes, expressions, filepath, channel_data,
                            standardizer):
    emax = expressions.max().max() * 1.05
    # ptmax = phenotypes.max().max() * 1.05
    ptmax = None
    # sorted_index = expressions[expressions.abs() >
    #                            0.3].count().sort_values(ascending=False).index
    sorted_index = phenotypes.columns
    name_dict = {
        tup.Index: tup.description
        for tup in channel_data.itertuples()
    }

    with PdfPages(filepath) as pdf:
        for pt in sorted_index:
            log.info(f'Plotting phenotype {pt}.')
            plot_phenotype(phenotypes[pt],
                           20,
                           expressions[pt],
                           phenotype_max=ptmax,
                           expression_max=emax,
                           name_descriptions=name_dict,
                           standardizer=standardizer)

            pdf.savefig()
            plt.close()


def plot_phenotype(pt,
                   n,
                   expressions,
                   plot_title=None,
                   phenotype_max=None,
                   expression_max=None,
                   name_descriptions=None,
                   standardizer=None):
    """Plot the top n values of phenotype Series pt.
    """
    colors = [
        'lightcoral', 'lightskyblue', 'palegreen', 'mediumvioletred',
        'slateblue', 'teal', 'indianred', 'cadetblue', 'mediumseagreen',
        'rosybrown', 'darkcyan', 'forestgreen'
    ]

    # modes are colored based on their order in the pt index, which in most
    # cases is the order that they are listed in the configs list of the
    # Experiment object.
    modes = pt.index.get_level_values(0).drop_duplicates()

    # only uses as many colors as needed.
    mode_color = dict(zip(modes, colors))

    plt.clf()
    f = plt.gcf()
    f.set_size_inches(11, 6)

    df = pd.DataFrame({
        'pt': pt,
        'ab': pt.abs()
    }).sort_values('ab', ascending=False)
    v = df.pt[:n]

    y = range(n - 1, -1, -1)
    plt.barh(y, v)
    if standardizer is None:
        label_func = lambda *unused: ''
    else:
        label_func = standardizer.inverse_transform_label

    if name_descriptions is None:
        names = v.index.get_level_values(1)
        modes = [None] * len(names)
    else:
        names = [
            '[{impact}] {desc}'.format(
                impact=f'{label_func(index_elt, v[index_elt])}',
                desc=name_descriptions.get(index_elt, 'No Description'))
            for index_elt in v.index
        ]
        modes = [index_elt[0] for index_elt in v.index]
    y_colors = [mode_color.get(m, 'white') for m in modes]
    if phenotype_max is None:
        xmin, xmax = plt.xlim()
        phenotype_max = max(abs(xmin), xmax)

    labels = [_truncate_string(name, 75) for name in names]
    plt.yticks(y, labels)
    plt.scatter([-0.95 * phenotype_max] * len(y),
                y,
                c=y_colors,
                marker='o',
                s=100,
                edgecolors='black',
                linewidths=0.5)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=True,  # ticks along the top edge are on
        labelbottom=False,  # labels along the bottom edge are on
        labeltop=True,  # labels along the top edge are on
        direction='in')

    plt.xlim((-phenotype_max, phenotype_max))
    plt.axvline(x=0)

    if not plot_title:
        plot_title = 'Phenotype ' + pt.name

    plt.title(plot_title, pad=10)
    _plot_inset(f, expressions, expression_max)

    plt.tight_layout()
    f.subplots_adjust(left=0.5, top=0.9)


def _plot_inset(fig, expressions, expression_max):
    left, bottom, width, height = [0.85, 0.11, 0.12, 0.20]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.set_alpha(0.4)
    ax2.hist(x=(expressions[expressions > 0], -expressions[expressions < 0]),
             bins=100,
             histtype='stepfilled',
             log=True,
             color=('blue', 'red'),
             alpha=0.4,
             label=('pos', 'neg'))
    ax2.legend(loc='upper right', fontsize='x-small', frameon=False)

    plt.xlabel('Expression')
    plt.ylabel('Count')
    ax2.patch.set_alpha(0.5)
    ax2.set_ylim(bottom=0.8, top=None)


def _truncate_string(s, n):
    if len(s) > n:
        return s[:n - 3] + '..'
    else:
        return s
