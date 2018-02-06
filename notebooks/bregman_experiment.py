import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
import glob
import cPickle as pkl

#from science_rcn.inference import test_image
#from science_rcn.learning import train_image
plt.ion()

def prep_img(img):
    img = imresize(img, (112,112))
    img = np.pad(img, pad_width=((44,44),(44,44)), mode='constant', constant_values=0)
    return img




# 
from itertools import izip
import logging
import networkx as nx
from numpy.random import rand, randint

from science_rcn.dilation.dilation import dilate_2d
from science_rcn.preproc import Preproc
from science_rcn.inference import forward_pass, LoopyBPInference


def densify(pos, img_shape):
    img = np.zeros(img_shape)
    search_space = {i: np.round((np.cos(np.pi * i / 8.), np.sin(np.pi * i / 8.)), 4)
                    for i in range(16)}

    for p in pos:
        img[p[1], p[2]] = 5.
        nearest_neighbor = p
        max_proj = -np.inf
        for q in pos:
            if np.all(p == q) or np.max(np.abs(p-q)) > 8:
                continue
            d = q[1:] - p[1:]
            d = d / np.sqrt(np.sum(d**2))
            if np.dot(d, search_space[p[0]]) > max_proj:
                nearest_neighbor = q
                max_proj = np.dot(d, search_space[p[0]])
        #print p, nearest_neighbor, max_proj
        d = nearest_neighbor[1:] - p[1:]
        for i in range(int(np.sqrt(np.sum(d**2)))):
            q = np.round(p[1:] + d*i/np.sqrt(np.sum(d ** 2)), 0).astype('int')
            img[q[0], q[1]] = 1.

    R = np.array([[0, 1], [-1, 0]])
    search_space = {
    i: np.dot(np.round((np.cos(np.pi * i / 8.), np.sin(np.pi * i / 8.)), 4), R)
    for i in range(16)}

    for p in pos:
        img[p[1], p[2]] = 5.
        nearest_neighbor = p
        max_proj = -np.inf
        for q in pos:
            if np.all(p == q) or np.max(np.abs(p - q)) > 20:
                continue
            d = q[1:] - p[1:]
            d = d / np.sqrt(np.sum(d ** 2))
            if np.dot(d, search_space[p[0]]) > max_proj:
                nearest_neighbor = q
                max_proj = np.dot(d, search_space[p[0]])
        #print p, nearest_neighbor, max_proj
        d = nearest_neighbor[1:] - p[1:]
        for i in range(int(np.sqrt(np.sum(d ** 2)))):
            q = np.round(p[1:] + d * i / np.sqrt(np.sum(d ** 2)), 0).astype(
                'int')
            img[q[0], q[1]] = 1.

    return img


def get_mask(img):
    from skimage import measure
    mask = measure.label(img, background=-1, connectivity=1)
    return mask > 1


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
    # return bp_scores, top_candidates, backtraces
    top_match = bp_scores[0], top_candidates[0], backtraces[0]
    dense_msg = densify(top_match[2], img.shape[:2])
    mask = get_mask(dense_msg)
    orig_bu_msg = bu_msg.copy()
    bu_msg[:, mask] = 0.3

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
        rcn_inf = LoopyBPInference(bu_msg, frcs, edge_factors, pool_shape,
                                   preproc_layer,
                                   n_iters=n_iters, damping=damping)
        score, bp_pos = rcn_inf.bwd_pass(return_backtrace=True)
        bp_scores.append(score)
        backtraces.append(bp_pos)
        if score >= winner_score:
            winner_idx, winner_score = (idx, score)

    next_top_match = bp_scores[0], top_candidates[0], backtraces[0]
    print "Detected with {}, {} with score {}".format(top_match[1] / 100,
                                                      next_top_match[1] / 100,
                                                      top_match[0] +
                                                      next_top_match[0])
    return top_match[0:2], next_top_match[:2]


if __name__ == '__main__':
    with open('/Users/swaroop/vicarious/science_rcn/data/MNIST/trained_model',
              'r') as f:
        model = pkl.load(f)

    # base = 'data/MNIST/testing/0/101.bmp'
    # occ = 'data/MNIST/testing/1/1004.bmp'
    base = 'data/MNIST/testing/0/101.bmp'
    occ = 'data/MNIST/testing/2/1002.bmp'
    img = imread(base)
    img = prep_img(img)
    test_image(img, model)
    img1 = imread(occ)
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    img1 = prep_img(img1)

    img_occ = img.copy()
    img_occ[img1 > 0] = img1[img1 > 0]
    plt.subplot(1, 3, 2)
    plt.imshow(img_occ)

    img_cut = img.copy()
    img_cut[img1 > 0] = 0
    plt.subplot(1, 3, 3)
    plt.imshow(img_cut)

    # Get multiple top backtraces and run BMF
    print test_image(img, model)
    print test_image(img_occ, model)
    print test_image(img_cut, model)
