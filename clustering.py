__author__ = 'tejas_khot'

import numpy as np
import theano.tensor as T 
from theano import function
import theano
import * from theano_alg

class Clustering:

    """
    This is the basic clustering class.
    """

    def __init__(self, mr, vr, sr, di, ce, node_id):
        """
        Initializes the clustering class.
        """

        self.common_init(mr, vr, sr, di, ce, node_id)

    def common_init(self, mr, vr, sr, di, ce, node_id):
        """
        Initialization function used by the base class and subclasses.

        @param mr: mean rate 
        @param vr: variance rate
        @param sr: starvation rate
        @param di: number of dimensions
        @param ce: number of centroids
        @param node_id: node id to uniquely identify node on each layer
        """

        self.MEANRATE = mr
        self.VARRATE = vr
        self.STARVRATE = sr
        self.DIMS = di
        self.CENTS = ce
        self.ID = node_id
        # srng = RandomStreams(seed=100)
        # rv_u = srng.uniform((self.CENTS, self.DIMS))
        # f = function([], rv_u)
        self.mean = self.floatX(np.random.rand(self.CENTS, self.DIMS))
        self.var = self.floatX(0.001 * np.ones((self.CENTS, self.DIMS)))
        self.starv = self.floatX(np.ones((self.CENTS, 1)))
        self.belief = self.floatX(np.zeros((1, self.CENTS)))
        self.children = []
        self.last = self.floatX(np.zeros((1, self.DIMS)))
        self.whitening = False

    def floatX(self, x):
        return np.asarray(x,dtype=theano.config.floatX)

    def update(self, input_, TRAIN):
        input_ = self.floatX(input_.reshape(1, self.DIMS))
        diff = theanoMatMatSub(input_, self.mean)
        sqdiff = diff**2
        if TRAIN:
            euc = T.sqrt(theanoMatSum(sqdiff, axs=1).reshape(self.CENTS, 1))
            # Apply starvation trace
            dist = theanoMatMatMul(dist, self.starv)
            # Find and Update Winner
            winner = dist.argmin()
            self.mean[winner, :] = theanoMatMatAdd(self.mean[winner, :], 
                                                    theanoMatMatMul(self.MEANRATE, diff[winner, :])
                                                   )
            # this should be updated to use sqdiff
            vdiff = theanoMatMatSub(diff[winner, :]**2 , self.var[winner, :])
            self.var[winner, :] = theanoMatMatAdd(self.var[winner, :], 
                                                  theanoMatMatMul(self.VARRATE, vdiff)
                                                  )
            self.starv *= (1.0 - self.STARVRATE)
            self.starv[winner] += self.STARVRATE

        normdist = theanoMatSum(theanoMatMatDiv(sqdiff, self.var), axs=1)
        chk = (normdist == 0)
        if any(chk):
            self.belief = np.zeros((1, self.CENTS))
            self.belief[chk] = 1.0
        else:
            normdist = theanoScaVecDiv(1, normdist)
            self.belief = theanoVecScaDiv(normdist, theanoVecSum(normdist)).reshape(1, self.CENTS)

    def add_child(self, child):
        """
        Add child node for providing input.

        @param child:
        """

        self.children.append(child)

    def latched_update(self, TRAIN):
        """
        Update node that has children nodes.

        @param TRAIN:
        """

        input_ = np.concatenate([c.belief for c in self.children], axis=1)
        self.update_node(input_, TRAIN)

    def init_whitening(self, mn=[], st=[], tr=[]):
        """
        Initialize whitening parameters.

        @param mn:
        @param st:
        @param tr:
        """

        # Rescale initial means, since inputs will no longer
        # be non-negative.
        self.mean[:, self.LABDIMS:self.LABDIMS + self.EXTDIMS] *= 2.0
        self.mean[:, self.LABDIMS:self.LABDIMS + self.EXTDIMS] -= 1.0

        self.whitening = True
        self.patch_mean = mn
        self.patch_std = st
        self.whiten_mat = tr

    def clear_belief(self):
        pass