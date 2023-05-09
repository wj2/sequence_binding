
import itertools as it
import numpy as np
import matplotlib.pyplot as plt

nov_data = {
    'first': {
        'dissimilar':(378, 636),
        'hue nonmatch':(236, 617),
        'dir nonmatch':(385,   630),
        'hue match':(393,   745),
        'dir match':(347,   596),
        'chimera hue match':(383,   751),
        'chimera dir match':(257,   611)
    },
    'second': {
        'dissimilar':(429,   589),
        'hue nonmatch': (378,   569),
        'dir nonmatch': (436,   609),
        'hue match': (359,   732),
        'dir match': (473,   664),
        'chimera hue match':(371,   738),
        'chimera dir match':(363, 535),
    },
}

corr_keys = tuple(it.product(('first', 'second'), ('dissimilar',)))
corr_all_keys = tuple(it.product(('first', 'second'),
                                 ('dissimilar', 'hue nonmatch', 'dir nonmatch',
                                  'hue match', 'dir match')))
nonchim_keys = tuple(it.product(('first', 'second'),
                                ('hue match', 'dir match')))
chim_keys = tuple(it.product(('first', 'second'),
                             ('chimera hue match', 'chimera dir match')))

def plot_conf_interval(data, x_val, keys=corr_keys, ax=None, alpha=.05,
                       fwid=2, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(fwid, fwid))
    num = 0
    denom = 0
    for (o_key, i_key) in keys:
        num = num + data[o_key][i_key][0]
        denom = denom + data[o_key][i_key][1]
    p = num/denom
    z = 1 - .5*alpha
    p_pm = z*np.sqrt(p*(1 - p)/denom)
    print(p_pm)
    l = ax.plot([x_val], p, 'o', **kwargs)
    color = l[0].get_color()
    ax.errorbar([x_val], p, yerr=p_pm, color=color)
