import numpy as np
import pickle
import treecorr
import galsim
import argparse
import pandas as pd
import pathlib


def raw_moment(arr, p_x, p_y):
    """
    Compute raw moments of arr
    """
    y, x = arr.shape
    yy, xx = np.meshgrid(np.arange(0, x, 1), np.arange(0, y, 1))

    return np.sum(np.power(xx, p_x) * np.power(yy, p_y) * arr)

def central_moment(a1, p_x, p_y):
    """
    Compute central moments of arr
    """
    y, x = a1.shape
    yy, xx = np.meshgrid(np.arange(0, x, 1), np.arange(0, y, 1))
    mu_0 = raw_moment(a1, 0, 0)
    mu_x = raw_moment(a1, 1, 0)
    mu_y = raw_moment(a1, 0, 1)
    return np.sum(np.power(xx - mu_x / mu_0, p_x) * np.power(yy - mu_y / mu_0, p_y) * a1)

def moment(arr, p_x, p_y):
    """
    Compute normalized central moments of arr
    """
    return central_moment(arr, p_x, p_y) / central_moment(arr, 0, 0)


def comp2pcfTreecorr(x, y, k, size='small', test_circle=False, bin_type='TwoD'):
    """Calculate 2pcf for scalar k."""
    if test_circle:
        rmax = 3.5/2
        keep = np.where(np.hypot(x, y) < rmax)[0]
        print(f'keeping {len(keep)/len(x) * 100:.2g}% of stars')
        cat = treecorr.Catalog(x=np.array(x)[keep], y=np.array(y)[keep], k=np.array(k)[keep], w=None)
    else:
        cat = treecorr.Catalog(x=x, y=y, k=k, w=None)

    if size=='big':
        mxsep = 1.5
        mnsep = 1e-4
    elif size=='small':
        mxsep = 0.15
        mnsep = 0 
    kk = treecorr.KKCorrelation(min_sep=mnsep, max_sep=mxsep, nbins=15,
                                bin_type=bin_type, bin_slop=0)
    kk.process(cat)
    return kk


def get2pcfParamsHSM(kk, circle=True):
    """Compute angle of PSF parameter 2pcf."""
    xi_img = kk.xi
    if circle:  # set pixels outside circle to 0
        xi_img[kk.rnom > kk.max_sep] = 0
    new_params = galsim.hsm.HSMParams(max_mom2_iter=2000)
    xi_moms = galsim.hsm.FindAdaptiveMom(galsim.Image(xi_img), hsmparams=new_params)

    return xi_moms.moments_sigma, xi_moms.observed_shape, xi_moms.moments_centroid


def get2pcfParamsUnweighted(kk, circle=True):
    """Compute parameters of 2pcf with unweighted moments."""
    xi_img = kk.xi
    if circle: # set pixels outside circle to 0
        xi_img[kk.rnom > kk.max_sep] = 0

    xi_xx = moment(xi_img, 2, 0)
    xi_yy = moment(xi_img, 0, 2)
    xi_xy = moment(xi_img, 1, 1)
    
    if xi_xx * x_yy < xi_xy**2:
        print('oh no!')
    sigmaSquared = np.sqrt(xi_xx * xi_yy - xi_xy**2)
    g1 = (xi_xx - xi_yy) / (xi_xx + xi_yy + 2 * sigmaSquared)
    g2 = 2 * xi_xy / (xi_xx + xi_yy + 2 * sigmaSquared)

    phi = np.arctan2(g2, g1) * 180 / np.pi
    gMag = np.hypot(g1, g2)
    q = (1 - gMag) / (1 + gMag)
    sigma = np.sqrt(sigmaSquared)

    return sigma, phi/2, q


def initSeeds():
    """Define seeds for the simulations."""
    seeds = list(range(6, 11)) + [22, 23, 25, 26, 27] + list(range(30, 40))
    seeds += list(range(42, 61)) + list(range(62, 171))
    return seeds


def getOutputSummary(d, pcfsize):
    """Load simulation data from a pickle file."""
    d_sigma = d['sigma'] - np.mean(d['sigma'])

    size, e1, e2 = {}, {}, {}

    for psf_param, p_sum in zip([d['e1'], d['e2'], d_sigma], [size, e1, e2]):
        twopcf = comp2pcfTreecorr(d['thx'], d['thy'], psf_param, pcfsize, test_circle=False)

        p_sum['sig'], p_sum['phi'], p_sum['q'] = get2pcfParamsUnweighted(twopcf)

        try:
            p_sum['sig_hsm'], p_sum['shape_hsm'], p_sum['centroid_hsm'] = get2pcfParamsHSM(twopcf)
            p_sum['flag_hsm'] = 0
            p_sum['error_hsm'] = np.nan
        except galsim.errors.GalSimHSMError as e:
            p_sum['sig_hsm'], p_sum['shape_hsm'], p_sum['centroid_hsm'] = 0,0,0 
            p_sum['flag_hsm'] = 1
            p_sum['error_hsm'] = e
    
        p_sum['autocorr'] = np.var(psf_param)
    return size, e1, e2



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", type=str)
    parser.add_argument("size", type=str)
    parser.add_argument("--outdir", type=str, default='/home/users/chebert/validate-psfws/summaries/')
    parser.add_argument("--simdir", type=str, default='/home/groups/burchat/mya')
    args = parser.parse_args()

    if args.kind == 'psfws':
        kind_path = 'sameh0_psfws/outh_psfws_'
    elif args.kind == 'rand':
        kind_path = 'sameh0_rand/outh_rand_'
    elif args.kind == 'randMatch':
        # kind_path = 'matchSpeedRand/outv_rand_'
        kind_path = 'matchProfileRand/outp_rand_'
    else:
        raise ValueError('kind input must be "psfws", "rand", "randMatch", or "randMatchProfile".')

    size_sum, e1_sum, e2_sum, atm_sum = {}, {}, {}, {}
    for s in initSeeds():
        fname = kind_path + f'{s}.pkl'
        file_path = pathlib.Path.joinpath(pathlib.Path(args.simdir), fname)
        try:
            d = pickle.load(open(file_path, 'rb'))
        except FileNotFoundError:
            print(f'Warning: file with seed {s} not found!!')

        size_sum[s], e1_sum[s], e2_sum[s] = getOutputSummary(d, args.size)
        atm_sum[s] = d['atmKwargs']

    for summary, name in zip([size_sum, e1_sum, e2_sum, atm_sum],
                             ['size', 'e1', 'e2', 'atm']):
        f_name = f'{name}_summary_{args.kind}_{args.size}.p'
        save_path = pathlib.Path.joinpath(pathlib.Path(args.outdir), f_name)
        pickle.dump(summary, open(save_path, 'wb'))
