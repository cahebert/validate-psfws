import numpy as np
import pickle
import argparse
import pandas as pd
import pathlib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", type=str)
    parser.add_argument("--nPool", type=int, default=1)
    parser.add_argument("--outdir", type=str, default='/home/users/chebert/validate-psfws/summaries/allSixMonths_shear/')
    parser.add_argument("--simdir", type=str, default='/home/groups/burchat/chebert/psfwsPaperSims/')
    args = parser.parse_args()

    if (args.kind == 'psfws') + (args.kind == 'rand') + (args.kind == 'match') != 1:
        raise ValueError('kind input must be "psfws", "rand", or "match"')

    N = 538

    def f(num):
        fname = args.kind + f'{num}.p'
        file_path = pathlib.Path.joinpath(pathlib.Path(args.simdir), fname)
        try:
            d = pickle.load(open(file_path, 'rb'))
        except FileNotFoundError:
            print(f'Warning: file with seed {s} not found!!')

        atmOut = d['atmKwargs']
        atmOut['atmSum'] = d['atmSummary']

        return num, atmOut

    atm = {}
    if args.nPool > 1:
        from multiprocessing import Pool
        with Pool(args.nPool) as pool:
            for out in pool.imap_unordered(f, range(N)):
                atm[out[0]] = out[1]
    else:
        for num in range(N):
            _, out = f(num)
            atm[num] = out[1]
    
    f_name = f'atm_polarSummary_{args.kind}.p'
    save_path = pathlib.Path.joinpath(pathlib.Path(args.outdir), f_name)
    pickle.dump(atm, open(save_path, 'wb'))
