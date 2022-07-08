import os
import time
import pickle

import galsim
import numpy as np
from scipy.optimize import bisect
from astropy.utils.console import ProgressBar

# utilities
def lodToDol(lst):
    keys = lst[0].keys()
    out = {}
    for k in keys:
        out[k] = []
    for l in lst:
        for k in keys:
            out[k].append(l[k])
    return out

# generate atm summary parameters
def generateInput(rng):
    ud = galsim.UniformDeviate(rng)
    filters = 'ugrizy'
    ifilter = int(ud()*6)
    filterName = filters[ifilter]

    airmass = ud()*(1.4-1.01)+1.01
    rawSeeing = ud()*(1.6-0.6)+0.6

    return filterName, airmass, rawSeeing

def vkSeeing(r0_500, wavelength, L0):
    kolm_seeing = galsim.Kolmogorov(r0_500=r0_500, lam=wavelength).fwhm
    r0 = r0_500 * (wavelength/500)**1.2
    arg = 1. - 2.183*(r0/L0)**0.356
    factor = np.sqrt(arg) if arg > 0.0 else 0.0
    return kolm_seeing*factor

def seeingResid(r0_500, wavelength, L0, targetSeeing):
    return vkSeeing(r0_500, wavelength, L0) - targetSeeing

def compute_r0_500(wavelength, L0, targetSeeing):
    """Returns r0_500 to use to get target seeing."""
    r0_500_max = min(1.0, L0*(1./2.183)**(-0.356)*(wavelength/500.)**1.2)
    r0_500_min = 0.01
    return bisect(seeingResid, r0_500_min, r0_500_max, args=(wavelength, L0, targetSeeing))

def genAtmSummary(rng):
    wlen_dict = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)
    filterName, airmass, rawSeeing = generateInput(rng)
    wavelength = wlen_dict[filterName]
    targetFWHM = rawSeeing * airmass**0.6 * (wavelength/500.)**(-0.3)
    # Draw L0 from truncated log normal
    gd = galsim.GaussianDeviate(rng)
    L0 = 0
    while L0 < 10.0 or L0 > 100:
        L0 = np.exp(gd() * 0.6 + np.log(25.0))
    # Compute r0_500 that yields targetFWHM
    r0_500 = compute_r0_500(wavelength, L0, targetFWHM)
    return {
        'filterName':filterName,
        'airmass':airmass,
        'rawSeeing':rawSeeing,
        'wavelength':wavelength,
        'targetFWHM':targetFWHM,
        'L0':L0,
        'r0_500':r0_500
    }

# generate realization parameters
def genAtmKwargs(rng, atmSummary, args):
    ud = galsim.UniformDeviate(rng)
    gd = galsim.GaussianDeviate(rng)

    altitudes = [0.0, 2.58, 5.16, 7.73, 12.89, 15.46]
    altitudes[0] = args.groundLayerHeight

    Eweights = np.array([0.652, 0.172, 0.055, 0.025, 0.074, 0.022])
    Uweights = np.array([1./6]*6)
    weights = (args.w-1)*Eweights + args.w*Uweights

    # randomize a bit
    weights = [np.abs(w*(1.0 + 0.1*gd())) for w in weights]
    weights = np.clip(weights, 0.01, 0.75)  # keep weights from straying too far.
    weights /= np.sum(weights)  # renormalize

    # Broadcast outer scale
    L0 = [atmSummary['L0']]*6
    speeds = [ud()*args.maxSpeed for _ in range(6)]
    directions = [ud()*360*galsim.degrees for _ in range(6)]
    atmKwargs = dict(
        r0_500=atmSummary['r0_500'], L0=L0, speed=speeds, direction=directions,
        altitude=altitudes, r0_weights=weights, screen_size=args.screen_size,
        screen_scale=args.screen_scale, rng=rng
    )
    return atmKwargs


if __name__ == '__main__':
    from argparse import ArgumentParser
    from multiprocessing import Pool
    parser = ArgumentParser()
    parser.add_argument('--atmSeed', type=int, default=1)
    parser.add_argument('--psfSeed', type=int, default=2)
    parser.add_argument('--npsf', type=int, default=10000)
    parser.add_argument('--nphot', type=int, default=int(1e6))
    parser.add_argument('--screen_size', type=float, default=819.2)
    parser.add_argument('--screen_scale', type=float, default=0.1)
    parser.add_argument('--fov', type=float, default=3.5, help="linear field of view in degrees")
    parser.add_argument('--groundLayerHeight', type=float, default=0.1)
    parser.add_argument('-w', type=float, default=0.0, help='weight interpolation factor')
    parser.add_argument('--maxSpeed', type=float, default=20.0, help='speed')
    parser.add_argument('--outdir', type=str, default='output')
    parser.add_argument('--outfile', type=str, default='out.pkl')
    parser.add_argument('--nPool', type=int, default=10)
    args = parser.parse_args()

    # Generate random atmospheric input statistics
    atmRng = galsim.BaseDeviate(args.atmSeed)
    atmSummary = genAtmSummary(atmRng)
    atmKwargs = genAtmKwargs(atmRng, atmSummary, args)

    print(atmKwargs['r0_weights'])
    atm = galsim.Atmosphere(**atmKwargs)
    aper = galsim.Aperture(
        diam=8.36, obscuration=0.61,
        lam=atmSummary['wavelength'], screen_list=atm
    )

    r0 = atmSummary['r0_500'] * (atmSummary['wavelength']/500)**1.2
    kcrit = 0.2
    kmax = kcrit / r0
    print("instantiating")
    atm.instantiate(kmax=kmax, check='phot')
    print("done")

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
    thxs *= args.fov
    thys *= args.fov
    psfPhotSeeds *= 2**20
    psfPhotSeeds = psfPhotSeeds.astype(np.int64)

    def f(aaaa):
        thx, thy, seed = aaaa
        rng = galsim.BaseDeviate(int(seed))
        theta = (thx*galsim.degrees, thy*galsim.degrees)
        psf = atm.makePSF(atmSummary['wavelength'], aper=aper, exptime=30.0, theta=theta)
        psf = galsim.Convolve(psf, galsim.Gaussian(fwhm=0.35))
        img = psf.drawImage(nx=50, ny=50, scale=0.2, method='phot', n_photons=args.nphot, rng=rng)
        mom = galsim.hsm.FindAdaptiveMom(img)
        return {
            'thx':thx,
            'thy':thy,
            'seed':seed,
            'x':mom.moments_centroid.x,
            'y':mom.moments_centroid.y,
            'sigma':mom.moments_sigma,
            'e1':mom.observed_shape.e1,
            'e2':mom.observed_shape.e2
        }

    output = []
    with Pool(args.nPool) as pool:
        with ProgressBar(args.npsf) as bar:
            for o in pool.imap_unordered(f, zip(thxs, thys, psfPhotSeeds)):
                output.append(o)
                bar.update()

    output = lodToDol(output)
    output['args'] = args
    output['atmSummary'] = atmSummary
    output['atmKwargs'] = atmKwargs

    fullpath = os.path.join(args.outdir, args.outfile)
    with open(fullpath, 'wb') as f:
        pickle.dump(output, f)
