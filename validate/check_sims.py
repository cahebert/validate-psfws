import numpy as np
import pickle
import os

simdir = '/home/groups/burchat/chebert/psfwsPaperSims/'
files = os.listdir(simdir)

randnums = [f.strip('rand').strip('.p') for f in files if 'rand' in f]
psfwsnums = [f.strip('psfws').strip('.p') for f in files if 'psfws' in f]
matchnums = [f.strip('match').strip('.p') for f in files if 'match' in f]

assert len(randnums) == len(psfwsnums)
assert len(randnums) == len(matchnums)

assert (np.sort(randnums) == np.sort(psfwsnums)).all()
assert (np.sort(randnums) == np.sort(matchnums)).all()

for num in randnums:
    dr = pickle.load(open(simdir + 'rand' + num + '.p', 'rb'))
    dp = pickle.load(open(simdir + 'psfws' + num + '.p', 'rb'))
    dm = pickle.load(open(simdir + 'match' + num + '.p', 'rb'))

    # check same psfws index use (only needed for matcn and psfws)
    assert dp['args'].i == dm['args'].i
    # check atmospheric seeds match
    assert dr['args'].atmSeed == dp['args'].atmSeed
    assert dr['args'].atmSeed == dm['args'].atmSeed
    # check psf seeds match
    assert dr['args'].psfSeed == dp['args'].psfSeed
    assert dr['args'].psfSeed == dm['args'].psfSeed
    # check that match and psfws have the same direction profile
    assert dr['atmKwargs']['direction'] == dr['atmKwargs']['direction']
