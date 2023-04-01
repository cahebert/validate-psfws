import os
import pickle

import galsim
import numpy as np
from scipy.optimize import bisect
import psfws


def lodToDol(lst):
    """Utilities."""
    keys = lst[0].keys()
    out = {}
    for k in keys:
        out[k] = []
    for bloop in lst:
        for k in keys:
            out[k].append(bloop[k])
    return out


def vkSeeing(r0_500, wavelength, L0):
    """Generate Von Karman FWHM from given r0 value and wavelength."""
    kolm_seeing = galsim.Kolmogorov(r0_500=r0_500, lam=wavelength).fwhm
    r0 = r0_500 * (wavelength/500)**1.2
    arg = 1. - 2.183*(r0/L0)**0.356
    factor = np.sqrt(arg) if arg > 0.0 else 0.0
    return kolm_seeing*factor


def seeingResid(r0_500, wavelength, L0, targetSeeing):
    """Produce residual between Von Karman FWHM and target."""
    return vkSeeing(r0_500, wavelength, L0) - targetSeeing


def compute_r0_500(wavelength, L0, targetSeeing):
    """Return r0_500 to use to get target seeing."""
    r0_500_max = min(1.0, L0*(1./2.183)**(-0.356)*(wavelength/500.)**1.2)
    r0_500_min = 0.01
    return bisect(seeingResid, r0_500_min, r0_500_max, args=(wavelength, L0, targetSeeing))


def genAtmSummary(rng):
    """Generate atmospheric statistics. Note should be SAME for all 3 sims."""
    # Draw L0 from truncated log normal
    gd = galsim.GaussianDeviate(rng)
    L0 = 0
    while L0 < 10.0 or L0 > 100:
        L0 = np.exp(gd() * 0.6 + np.log(25.0))

    ud = galsim.UniformDeviate(rng)
    rawSeeing = ud()*(1.6-0.6)+0.6
    wavelength = 622.20  # r band
    airmass = 1  # at zenith
    targetFWHM = rawSeeing * airmass**0.6 * (wavelength/500.)**(-0.3)
    r0_500 = compute_r0_500(wavelength, L0, targetFWHM)

    return {'filterName': 'r',
            'rawSeeing': rawSeeing,
            'wavelength': wavelength,
            'targetFWHM': targetFWHM,
            'L0': L0,
            'r0_500': r0_500}


def setScreenSize(speeds, args):
    """Set screen size according to what test is running."""
    vmax = np.max(speeds)

    if args.testtype == 'full':
        #if vmax > 45:
        #    screen_size = 1600
        #else:
        screen_size = vmax * 30
    else:
        screen_size = vmax * 10

    return screen_size


def genAtmKwargs(rng, atmSummary, args):
    """Generate input for galsim.atmosphere using psfws."""
    L0 = [atmSummary['L0']] * 6

    ws = psfws.ParameterGenerator(seed=args.atmSeed)
    params = ws.draw_parameters(nl=6, location='com')
    params['h'] = [p - ws.h0 for p in params['h']]
    params['h'][0] += args.groundLayerHeight
    params['phi'] = [p * galsim.degrees for p in params['phi']]

    atmKwargs = dict(r0_500=atmSummary['r0_500'],
                     L0=L0,
                     speed=params['speed'],
                     direction=params['phi'],
                     altitude=params['h'],
                     r0_weights=params['j'],
                     screen_size=setScreenSize(params['speed'], args),
                     screen_scale=args.screen_scale,
                     rng=rng)
    return atmKwargs


if __name__ == '__main__':
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from multiprocessing import Pool
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('--atmSeed', type=int, default=1,
                        help="Random seed for generating atmosphere stats")
    parser.add_argument('--psfSeed', type=int, default=2,
                        help="Random seed for generating PSF stats")
    parser.add_argument('--npsf', type=int, default=10000,
                        help="Number of PSF simulated")
    parser.add_argument('--nphot', type=int, default=int(1e6),
                        help="Number of photons")
    parser.add_argument('--screen_scale', type=float, default=0.1,
                        help="Resolution of atmospheric screen in meters")
    parser.add_argument('--groundLayerHeight', type=float, default=0.2,
                        help="Ground layer height.")
    parser.add_argument('--outdir', type=str, default='/home/groups/burchat/chebert/')
    parser.add_argument('--outfile', type=str, default='outpsfws.pkl')
    parser.add_argument('--nPool', type=int, default=10,
                        help="Number of branches for parrallelization?")
    parser.add_argument('--testtype', type=str, default='full',
                        help="full, split, or wrap.")
    args = parser.parse_args()

    if args.testtype == 'split':
        expTime = 10.0
    else:
        expTime = 30.0

    # Generate random atmospheric input statistics
    atmRng = galsim.BaseDeviate(args.atmSeed)
    atmSummary = genAtmSummary(atmRng)
    atmKwargs = genAtmKwargs(atmRng, atmSummary, args)
    lam = atmSummary['wavelength']

    atm = galsim.Atmosphere(**atmKwargs)
    aper = galsim.Aperture(diam=8.36, obscuration=0.61, lam=lam, screen_list=atm)

    r0 = atmSummary['r0_500'] * (lam/500)**1.2
    kcrit = 0.2
    kmax = kcrit / r0
    atm.instantiate(kmax=kmax, check='phot')

    if args.testtype == 'split':
        atmSplit = []
        aperSplit = []

        for i in range(2):
            atmKwargs['rng'].discard(100)
            atmSplit.append(galsim.Atmosphere(**atmKwargs))
            aperSplit.append(galsim.Aperture(diam=8.36, obscuration=0.61,
                                             lam=lam, screen_list=atmSplit[i]))
            atmSplit[i].instantiate(kmax=kmax, check='phot')

    psfRng = galsim.BaseDeviate(args.psfSeed)
    ud = galsim.UniformDeviate(psfRng)
    thxs = np.empty(args.npsf, dtype=float)
    thys = np.empty(args.npsf, dtype=float)
    psfPhotSeeds = np.empty(args.npsf, dtype=float)
    ud.generate(thxs)
    ud.generate(thys)
    ud.generate(psfPhotSeeds)
    thxs -= 0.5
    thys -= 0.5
    thxs *= 3.5  # fov = 3.5 degrees
    thys *= 3.5  # fov = 3.5 degrees
    psfPhotSeeds *= 2**20
    psfPhotSeeds = psfPhotSeeds.astype(np.int64)

    def f(aaaa):
        thx, thy, seed = aaaa
        rng = galsim.BaseDeviate(int(seed))
        theta = (thx*galsim.degrees, thy*galsim.degrees)

        psf = atm.makePSF(lam, aper=aper, exptime=expTime, theta=theta)
        psf = galsim.Convolve(psf, galsim.Gaussian(fwhm=0.35))
        img = psf.drawImage(nx=50, ny=50, scale=0.2, method='phot',
                            n_photons=args.nphot, rng=rng)
        if args.testtype == 'split':
            img = img.array
            for i in range(2):
                psf = atmSplit[i].makePSF(lam, aper=aperSplit[i],
                                          exptime=expTime, theta=theta)
                psf = galsim.Convolve(psf, galsim.Gaussian(fwhm=0.35))
                img += psf.drawImage(nx=50, ny=50, scale=0.2, method='phot',
                                     n_photons=args.nphot, rng=rng).array
            img = galsim.Image(img)

        mom = galsim.hsm.FindAdaptiveMom(img)
        return {'thx': thx,
                'thy': thy,
                'seed': seed,
                'x': mom.moments_centroid.x,
                'y': mom.moments_centroid.y,
                'sigma': mom.moments_sigma,
                'e1': mom.observed_shape.e1,
                'e2': mom.observed_shape.e2}

    output = []
    with Pool(args.nPool) as pool:
        for o in pool.imap_unordered(f, zip(thxs, thys, psfPhotSeeds)):
            output.append(o)

    output = lodToDol(output)
    output['args'] = args
    output['atmKwargs'] = atmKwargs

    fullpath = os.path.join(args.outdir, args.outfile)
    with open(fullpath, 'wb') as f:
        pickle.dump(output, f)

