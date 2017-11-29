import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
import pylab as plt
import glob
import cPickle as pkl

from science_rcn.inference import test_image
from science_rcn.learning import train_image
plt.ion()

def prep_img(img):
	img = imresize(img, (112,112))
	img = np.pad(img, pad_width=((44,44),(44,44)), mode='constant', constant_values=0)
	return img


with open('/Users/swaroop/vicarious/science_rcn/data/MNIST/trained_model', 'r') as f:
    model = pkl.load(f)

#base = 'data/MNIST/testing/0/101.bmp'
#occ = 'data/MNIST/testing/1/1004.bmp'
base = 'data/MNIST/testing/0/101.bmp'
occ = 'data/MNIST/testing/2/1002.bmp'
img = imread(base)
img = prep_img(img)
test_image(img, model)
img1 = imread(occ)
img1 = prep_img(img1)

img_occ = img.copy()
img_occ[img1>0] = img1[img1>0]
plt.imshow(img_occ)

img_cut = img.copy()
img_cut[img1>0] = 0
plt.imshow(img_cut)

# Get multiple top backtraces and run BMF
print test_image(img, model)
print test_image(img_occ, model)
print test_image(img_cut, model)


# 
from itertools import izip
import logging
import networkx as nx
from numpy.random import rand, randint

from science_rcn.dilation.dilation import dilate_2d
from science_rcn.preproc import Preproc
from science_rcn.inference import forward_pass, LoopyBPInference

def test_image(img, model_factors,
               pool_shape=(25, 25), num_candidates=20, n_iters=300, damping=1.0):
    """
    Main function for testing on one image.

    Parameters
    ----------
    img : 2D numpy.ndarray
        The testing image.
    model_factors : ([numpy.ndarray], [numpy.ndarray], [networkx.Graph])
        ([frcs], [edge_factors], [graphs]), output of train_image in learning.py.
    pool_shape : (int, int)
        Vertical and horizontal pool shapes.
    num_candidates : int
        Number of top candidates for backward-pass inference.
    n_iters : int
        Maximum number of loopy BP iterations.
    damping : float
        Damping parameter for loopy BP.

    Returns
    -------
    winner_idx : int
        Training index of the winner feature.
    winner_score : float
        Score of the winning feature.
    """
    # Get bottom-up messages from the pre-processing layer
    preproc_layer = Preproc(cross_channel_pooling=True)
    bu_msg = preproc_layer.fwd_infer(img)

    # Forward pass inference
    fp_scores = np.zeros(len(model_factors[0]))
    for i, (frcs, _, graph) in enumerate(izip(*model_factors)):
        fp_scores[i] = forward_pass(frcs,
                                    bu_msg,
                                    graph,
                                    pool_shape)
    top_candidates = np.argsort(fp_scores)[-num_candidates:]

    # Backward pass inference
    winner_idx, winner_score = (-1, -np.inf)  # (training feature idx, score)
    bp_scores = []
    backtraces = []
    for idx in top_candidates:
        frcs, edge_factors = model_factors[0][idx], model_factors[1][idx]
        rcn_inf = LoopyBPInference(bu_msg, frcs, edge_factors, pool_shape, preproc_layer,
                                   n_iters=n_iters, damping=damping)
        score, bp_pos = rcn_inf.bwd_pass(return_backtrace=True)
        bp_scores.append(score)
        backtraces.append(bp_pos)
        if score >= winner_score:
            winner_idx, winner_score = (idx, score)
    #return winner_idx, winner_score, 
    return bp_scores, top_candidates, backtraces