import os
import time
import pickle

import galsim
import numpy as np
from scipy.optimize import bisect
import psfws

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

if __name__ == '__main__':
    from argparse import ArgumentParser, RawDescriptionHelpFormatter, BooleanOptionalAction
    from multiprocessing import Pool
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('--psfSeed', type=int, default=2,
            help="Random seed for generating PSF stats")
    parser.add_argument('--npsf', type=int, default=10000,
            help="Number of PSF simulated")
    parser.add_argument('--nphot', type=int, default=int(1e6),
            help="Number of photons")
    parser.add_argument('--fov', type=float, default=3.5, 
            help="linear field of view in degrees")
    parser.add_argument('--outdir', type=str, default='output')
    parser.add_argument('--outfile', type=str, default='outpsfws.pkl')
    parser.add_argument('--nPool', type=int, default=10,
            help="Number of branches for parrallelization?")
    args = parser.parse_args()

    wavelength = 754.06 #622.20 
    psfRng = galsim.BaseDeviate(args.psfSeed)
    ud = galsim.UniformDeviate(psfRng)
    # fwhm from 0.6 to 1.6 arcsec = (2.9,7.76)e-6 radians 
    # -> r0 varies from (.21,.078)m (without taking L0 into account)  
    r0 = ud()*(0.21-0.078)+0.078
    stdE = pickle.load(open('/home/users/chebert/validate-psfws/summaries/allSixMonths_shear/avgStdE_psfws.p','rb'))

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
        psf = galsim.Kolmogorov(r0=r0, lam=wavelength)
        g = galsim.GaussianDeviate(rng, mean=0, sigma=stdE)
        e = abs(g())
        if e > 0.8: e -= 0.2
        u = galsim.UniformDeviate(rng)
        psf = psf.shear(g=e, beta=u()*180*galsim.degrees)
        #convolving atm psf with gaussian optical psf
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
        for o in pool.imap_unordered(f, zip(thxs, thys, psfPhotSeeds)):
            output.append(o)

    output = lodToDol(output)
    fullpath = os.path.join(args.outdir, args.outfile)
    with open(fullpath, 'wb') as f:
        pickle.dump(output, f)
