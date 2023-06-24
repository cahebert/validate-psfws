import numpy as np
import pickle
import argparse
import pathlib

def getOutputE(d):
    """Extract std(|e|) from simulation output file."""
    eMag = np.hypot(d['e1'], d['e2'])
    return np.std(eMag)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nPool", type=int, default=1)
    parser.add_argument("--outdir", type=str, default='/home/users/chebert/validate-psfws/summaries/allSixMonths_shear/')
    parser.add_argument("--simdir", type=str, default='/home/groups/burchat/chebert/psfwsPaperSims/')
    args = parser.parse_args()

    kind = 'psfws'

    def f(num):
        fname = kind + f'{num}.p'
        file_path = pathlib.Path.joinpath(pathlib.Path(args.simdir), fname)
        try:
            d = pickle.load(open(file_path, 'rb'))
        except FileNotFoundError:
            print(f'Warning: file with seed {num} not found!!')

        return num, getOutputE(d)

    stdE = {}
    if args.nPool > 1:
        from multiprocessing import Pool
        with Pool(args.nPool) as pool:
            for out in pool.imap_unordered(f, range(N+1)):
                num = out[0]
                stdE[num] = out[1]
    else:
        for num in range(538):
            _, out = f(num)
            stdE[num] = out

    f_name = f'avgStdE_{kind}.p'
    save_path = pathlib.Path.joinpath(pathlib.Path(args.outdir), f_name)
    avgStdE = np.mean([se for k,se in stdE.items()])
    pickle.dump(avgStdE, open(save_path, 'wb'))
