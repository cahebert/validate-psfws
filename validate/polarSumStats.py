import numpy as np
import pickle
import treecorr
import galsim
import argparse
import pandas as pd
import pathlib
from collections import namedtuple


def comp2pcfTreecorr(x, y, k, size):
    """Calculate 2pcf for scalar k."""
    cat = treecorr.Catalog(x=x, y=y, k=k, w=None)
    if size=='big':
        mxsep = 2
        nbins = 101
        mnsep = 0.1
    elif size=='small':
        mxsep = 0.075
        nbins = 31
        mnsep = 0
    kk = treecorr.KKCorrelation(min_sep=mnsep, max_sep=mxsep, nbins=nbins,
                                bin_type='TwoD', bin_slop=0)
    kk.process(cat)
    return kk


def get2pcfPolar(kk):
    """Cast PSF parameter 2pcf into polar coordinates."""
    dx = np.linspace(-kk.max_sep, kk.max_sep, kk.nbins)
    dy = np.linspace(-kk.max_sep, kk.max_sep, kk.nbins)
    dX, dY = np.meshgrid(dx, dy)
    r = np.hypot(dX, dY).ravel()
    theta = np.arctan2(dY, dX).ravel() * 180 / np.pi
    keep = (theta > 0)
    theta = 90 - theta
    polar = namedtuple('polar', 'r theta xi')
    return polar(r[keep], theta[keep], kk.xi.ravel()[keep])


def get2pcfAnnulus(kkPol, radius, width):
    """Return annulus of radius and width from 2pcf."""
    inRing = (kkPol.r > radius) & (kkPol.r < radius + width)
    annulus = namedtuple('annulus', 'r theta xi')
    return annulus(kkPol.r[inRing], kkPol.theta[inRing], kkPol.xi[inRing])


def getHistogram(annulus, nBins):
    """Return a binned version of given annulus."""
    theEdges = np.linspace(-90, 90, nBins + 1)
    theBins = (theEdges[1:] + theEdges[:-1])/2
    conditions = [(annulus.theta>=theEdges[i])&(annulus.theta<theEdges[i+1])
                  for i in range(nBins)]
    binnedXi = [np.mean(annulus.xi[conditions[i]]) for i in range(nBins)]
    
    # want to convert this to bins from 0 to 180 for display purposes
    posBins = (theBins+180)%180
    order = np.argsort(posBins)

    return np.array(binnedXi)[order], posBins[order]

def initSeeds():
    """Define seeds for the simulations."""
    seeds = list(range(6, 11)) + [22, 23, 25, 26, 27] + list(range(30, 40))
    seeds += list(range(42, 61)) + list(range(62, 171))
    return seeds


def getOutputSummary(d, outputCoords=False):
    """Extract summary parameters from simulation output file."""
    d_sigma = d['sigma'] - np.mean(d['sigma'])

    size, e1, e2 = {}, {}, {}
    imgs = []
    coords = {}
    
    for psf_param, p_sum in zip([d['e1'], d['e2'], d_sigma], [e1, e2, size]):
        for pcf_size, radius, width, nbins in zip(['big', 'small'], [1.75, 0.05], [0.25, 0.025], [30, 18]):
            kk = comp2pcfTreecorr(d['thx'], d['thy'], psf_param, pcf_size)
            kkPolar = get2pcfPolar(kk)
            kkAnnulus = get2pcfAnnulus(kkPolar, radius, width)        
            p_sum['kkSlice_'+pcf_size], p_sum['kkSliceBins_'+pcf_size] = getHistogram(kkAnnulus, nbins)            

            imgs.append(kkPolar.xi)
            if outputCoords and p_sum == size:
                coords[pcf_size+'_theta'] = kkPolar.theta
                coords[pcf_size+'_r'] = kkPolar.r
        p_sum['autocorr'] = np.var(psf_param)
    return size, e1, e2, imgs, coords


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", type=str)
    parser.add_argument("--outdir", type=str, default='/home/users/chebert/validate-psfws/summaries/')
    parser.add_argument("--simdir", type=str, default='/home/groups/burchat/mya')
    args = parser.parse_args()

    if args.kind == 'psfws':
        kind_path = 'sameh0_psfws/outh_psfws_'
    elif args.kind == 'rand':
        kind_path = 'sameh0_rand/outh_rand_'
    elif args.kind == 'match':
        kind_path = 'randMatchDir/out_matchDir_'
    else:
        raise ValueError('kind input must be "psfws", "rand", or "match".')

    size_sum, e1_sum, e2_sum, atm_sum = {}, {}, {}, {}
    images = {}
    for i, s in enumerate(initSeeds()):
        fname = kind_path + f'{s}.pkl'
        file_path = pathlib.Path.joinpath(pathlib.Path(args.simdir), fname)
        try:
            d = pickle.load(open(file_path, 'rb'))
        except FileNotFoundError:
            print(f'Warning: file with seed {s} not found!!')

        size_sum[s], e1_sum[s], e2_sum[s], imgs, coords = getOutputSummary(d, outputCoords=True)
        atm_sum[s] = d['atmKwargs']

        images[s] = {k:imgs[i] for i,k in enumerate(['e1_big', 'e1_small', 'e2_big', 'e2_small', 'size_big', 'size_small'])}
    images['coord'] = coords    

    f_name = f'polarSummary_images_{args.kind}.p'
    save_path = pathlib.Path.joinpath(pathlib.Path(args.outdir), f_name)
    pickle.dump(images, open(save_path, 'wb'))

    for summary, name in zip([size_sum, e1_sum, e2_sum, atm_sum],
                             ['size', 'e1', 'e2', 'atm']):
        f_name = f'{name}_polarSummary_{args.kind}.p'
        save_path = pathlib.Path.joinpath(pathlib.Path(args.outdir), f_name)
        pickle.dump(summary, open(save_path, 'wb'))
