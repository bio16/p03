#!/usr/bin/env python
# -*- coding: utf-8 -*-
import igraph
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np
import math
from pylab import figure, close
import matplotlib.cm as cm

def contour_2d(fig, ax, x, y, mat, hscale='log', **kargs):
    cb_label = kargs.get('cb_label', 'points per bin square')
    cb_fontsize = kargs.get('cb_fontsize', 15)
    cbmin, cbmax = kargs.get('vmin',1), kargs.get('vmax',1e3)
    opt = {
    'linewidth': 0,
    'cmap': cm.gray_r,                # gray-scale
    'vmin': cbmin, #kargs.get('cbmin',1),
    'vmax': cbmax, #kargs.get('cbmax',1000),
    'alpha': kargs.get('alpha',0.9),
    }
    if hscale=='log':
        opt.update({'norm': LogNorm(),})
    #--- 2d contour
    surf = ax.contourf(x, y, mat, facecolors=cm.jet(mat), **opt)
    sm = cm.ScalarMappable(cmap=surf.cmap, norm=surf.norm)
    sm.set_array(mat)
    #--- colorbar
    axcb = fig.colorbar(sm)
    axcb.set_label(cb_label, fontsize=cb_fontsize)
    sm.set_clim(vmin=cbmin, vmax=cbmax)
    return fig, ax


def fprobs(g, set_sex, comms, aname='greedy'):
    """
    - g:
    graph
    - comms:
    dict containing the communities detected by different 
    algorithms.
    - aname:
    algorithm name
    """
    if not aname in ('louvain','infomap'):
        clustering = comms[aname].as_clustering()
        membership = clustering.membership
    else:
        membership = comms[aname].membership

    set_memb = np.array(list(set(membership)))
    # frequentist probability
    fprob = np.zeros(shape=(set_sex.size,set_memb.size), dtype=np.float)
    for i in range(len(g.vs)):
        g.vs[i]["membership"] = membership[i]

    # let's sample both labels && accumulate into `fprob`
    for v in g.vs:
        # catch the membership-index 
        i_m = (v['membership']==set_memb).nonzero()[0][0]
        # catch the sex-index 
        i_s = (v['sex']==set_sex).nonzero()[0][0]
        fprob[i_s,i_m] += 1.0
    # normalize 
    fprob /= fprob.sum()
    # return conj && marginates
    fp = {
    'conj'       : fprob,
    'sex'        : np.sum(fprob, axis=1),
    'membership' : np.sum(fprob, axis=0),
    }
    return fp

def information(p12, p1, p2):
    """
    we are using:
    I({C1},{C2}) = \sum_{C1,C2} P(C1,C2) * log(P(C1,C2)/(P(C1)*P(C2)))
    """
    # log term
    log_pp = np.log(p12) - np.log(np.outer(p1,p2))
    I = 0.0 # total mutual information
    for i_s in range(p12.shape[0]):
        for i_m in range(p12.shape[1]):
            #NOTE: contribute to sum only if joint probability is >0.0
            I += p12[i_s,i_m]*log_pp[i_s,i_m] if p12[i_s,i_m]>0.0 else 0.0
    return I
