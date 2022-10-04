import numpy as np
import pickle
import treecorr
import galsim
import argparse
import pandas as pd
import pathlib


def comp2pcfTreecorr(x, y, k, bin_type='TwoD'):
    """Calculate 2pcf for scalar k."""
    cat = treecorr.Catalog(x=x, y=y, k=k, w=None)
    kk = treecorr.KKCorrelation(min_sep=0, max_sep=0.15, nbins=17,
                                bin_type=bin_type, bin_slop=0)
    kk.process(cat)
    return kk


def get2pcfParams(kk, circle=True):
    """Compute angle of PSF parameter 2pcf."""
    xi_img = kk.xi
    if circle:  # set pixels outside circle to 0
        xi_img[kk.rnom > kk.max_sep] = 0
    new_params = galsim.hsm.HSMParams(max_mom2_iter=1500)
    xi_moms = galsim.hsm.FindAdaptiveMom(galsim.Image(xi_img), hsmparams=new_params)

    phi = np.arctan2(xi_moms.observed_shape.g2, xi_moms.observed_shape.g1)
    phi *= 180 / np.pi
    if phi < 0:
        phi += 360

    return phi / 2, xi_moms.moments_sigma


def initSeeds():
    """Define seeds for the simulations."""
    seeds = list(range(6, 11)) + [22, 23, 25, 26, 27] + list(range(30, 40))
    seeds += list(range(42, 171))
    return seeds


def getOutputSummary(d):
    """Load simulation data from a pickle file."""
    d_sigma = d['sigma'] - np.mean(d['sigma'])

    size, e1, e2 = {}, {}, {}

    for psf_param, p_sum in zip([d['e1'], d['e2'], d_sigma], [size, e1, e2]):
        twopcf = comp2pcfTreecorr(d['thx'], d['thy'], psf_param)

    # try:
    #    p_sum['2p_dir_circ'], p_sum['2p_sig_circ'] = get2pcfParams(twopcf)
    # except galsim.errors.GalSimHSMError:
    #    print(f'moments estimation failed!')
        try:
            p_sum['2p_dir'], p_sum['2p_sig'] = get2pcfParams(twopcf, circle=False)
        except galsim.errors.GalSimHSMError:
            print(f'moments estimation failed!')
            p_sum['2p_dir'], p_sum['2p_sig'] = 0,0 
        p_sum['autocorr'] = np.var(psf_param)

    return size, e1, e2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", type=str)
    parser.add_argument("--outdir", type=str, default='../summaries/')
    parser.add_argument("--simdir", type=str, default='/home/groups/burchat/mya')
    args = parser.parse_args()

    if args.kind == 'psfws':
        kind_path = 'sameh0_psfws/outh_psfws_'
    elif args.kind == 'rand':
        kind_path = 'sameh0_rand/outh_rand_'
    elif args.kind == 'randMatch':
        kind_path = 'matchSpeedRand/outv_rand_'
    else:
        raise ValueError('kind input must be "psfws", "rand", or "randMatch".')

    size_sum, e1_sum, e2_sum, atm_sum = [], [], [], []
    for s in initSeeds():
        fname = kind_path + f'{s}.pkl'
        file_path = pathlib.Path.joinpath(pathlib.Path(args.simdir), fname)
        try:
            d = pickle.load(open(file_path, 'rb'))
        except FileNotFoundError:
            print(f'Warning: file with seed {s} not found!!')

        out_summaries = getOutputSummary(d)
        size_sum.append(out_summaries[0])
        e1_sum.append(out_summaries[1])
        e2_sum.append(out_summaries[2])
        atm_sum.append(d['atmKwargs'])

    for summary, name in zip([size_sum, e1_sum, e2_sum, atm_sum],
                             ['size', 'e1', 'e2', 'atm']):
        df = pd.DataFrame(data=summary, index=initSeeds())
        f_name = f'{name}_summary_{args.kind}_df.p'
        save_path = pathlib.Path.joinpath(pathlib.Path(args.outdir), f_name)
        pickle.dump(df, open(save_path, 'wb'))