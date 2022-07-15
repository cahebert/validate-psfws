import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d


def fig_config(fig, ax, xlabel, ylabel):
    """
    setting labels for plots
    """
    ax.set_xlabel(xlabel, size=10, labelpad=3)
    ax.set_ylabel(ylabel, size=10)
    # ax.set_title(title, size=15, pad=-50)


def make_grid(fig, gs):
    """
    making grids for the image
    """
    gs = fig.add_gridspec(nrows=3, ncols=3, width_ratios=[1.5, 2, 2], height_ratios=[2, 2, 1], hspace=0.3)
    # make grid
    gs_hexbin = gs[0, 1:].subgridspec(1, 2, wspace=0.3)
    gs_new = gs[1, 1:].subgridspec(1, 2, wspace=0.2)
    gs_hist = gs[2, 1:].subgridspec(1, 3, wspace=0.3)
    gs_wind = gs[:, 0].subgridspec(5, 1, hspace=0.2)
    return gs_hexbin, gs_new, gs_hist, gs_wind


def hex_psf(fig, ax_hexPsf, thx, thy, sigma):
    """
    2D hexbin plot of psf position with psf size as weight
    """
    im0 = ax_hexPsf.hexbin(thx, thy, gridsize=30, C=sigma, cmap="inferno", alpha=0.75)
    fig_config(fig, ax_hexPsf, "$\Theta_x$", "$\Theta_y$")
    ax_hexPsf.margins(0.1, 0.1)
    divider0 = make_axes_locatable(ax_hexPsf)
    cax0 = divider0.append_axes('top', size='5%', pad=0.1)
    cbr0 = plt.colorbar(im0, cax=cax0, orientation='horizontal').set_label("$\sigma$")
    cax0.xaxis.set_ticks_position("top")
    cax0.xaxis.set_label_position("top")
    return ax_hexPsf


def hex_shear(fig, ax_hexSh, thx, thy, e1, e2):
    """
    2D hexbin plot of psf position with shear magnitude as weight
    """
    im1 = ax_hexSh.hexbin(thx, thy, gridsize=30, C=np.hypot(e1, e2), cmap="BuPu")
    fig_config(fig, ax_hexSh, "$\Theta_x$", "$\Theta_y$")
    ax_hexSh.margins(0.1, 0.1)
    divider1 = make_axes_locatable(ax_hexSh)
    cax1 = divider1.append_axes('top', size='5%', pad=0.1)
    cbr1 = plt.colorbar(im1, cax=cax1, orientation='horizontal').set_label("|e|")
    cax1.xaxis.set_ticks_position("top")
    cax1.xaxis.set_label_position("top")
    return ax_hexSh


def plot_whisker(ax, thx, thy, e1, e2):
    """
    2D vector field plot of shear
    """
    e = np.hypot(e1, e2)
    beta = 0.5 * np.arctan2(e2, e1)
    dx = e * np.cos(beta)
    dy = e * np.sin(beta)
    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.9, 1.9)
    ax.set_xlabel('[degree]', labelpad=-2, fontsize=10)

    qdict = dict(
        alpha=1,
        angles='uv',
        headlength=0,
        headwidth=0,
        headaxislength=0,
        minlength=0,
        pivot='middle',
        width=0.0025,
        cmap="BuPu"
    )
    q = ax.quiver(thx, thy, dx, dy, e, scale=1.0, **qdict)
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes("top", size="4%", pad="2%")
    cbar = plt.colorbar(q, cax=ax_cb, orientation='horizontal').set_label(label='|e|',
                                                                          fontsize=12)
    ax_cb.xaxis.set_ticks_position("top")
    ax_cb.xaxis.set_label_position("top")
    return ax


def plt_hist(fig, ax, para, colour, xlabel, ylabel):
    """
    plot histogram
    """
    ax.hist(para, color=colour, bins=20)
    fig_config(fig, ax, xlabel, ylabel)
    return ax


def plt_wind(fig, ax, layers, para, x, clr, xlabel, ylabel):
    """
    scatter plot with interpolated lines
    """
    f = interp1d(layers, para, kind="cubic")
    ax.scatter(layers, para, color=clr)
    ax.plot(x, f(x), color=clr)
    fig_config(fig, ax, xlabel, ylabel)
    return ax


def plot_results(args):
    data = pickle.load(open(os.path.join(args.outdir, args.outfile), 'rb'))
    spd = data['atmKwargs']["speed"]
    layers = data['atmKwargs']['altitude']
    direc = data['atmKwargs']["direction"]
    direc = [direc[i].deg for i in range(len(direc))]
    r0w = data['atmKwargs']["r0_weights"]
    xi = np.linspace(layers[0], layers[-1], 50)

    for key in data.keys():
        data[key] = np.array(data[key])
    (thx, thy, seed, x, y, sigma, e1, e2, arguments, atmSummary, atmKwargs) = tuple(data.values())


    fig = plt.figure(figsize=(18,17))
    #fig.suptitle("Original parameters: 1e4 PSFs, atmSeed 1, psfSeed 2", size=18, labelpad=-15)
    fig.set_facecolor("white")

    # make grid
    gs = fig.add_gridspec(nrows=3, ncols=3, width_ratios=[1.5,2,2], height_ratios=[2,2,1], hspace=0.3)
    gs_hexbin, gs_new, gs_hist, gs_wind = make_grid(fig, gs)

    # add plot to grid
    ax_hexPsf = fig.add_subplot(gs_hexbin[0])
    ax_hexSh = fig.add_subplot(gs_hexbin[1])

    ax_whisker = fig.add_subplot(gs_new[0])
    ax_his2d = fig.add_subplot(gs_new[1])

    ax_hisPsf = fig.add_subplot(gs_hist[0])
    ax_hise1 = fig.add_subplot(gs_hist[1])
    ax_hise2 = fig.add_subplot(gs_hist[2])

    ax_r0 = fig.add_subplot(gs_wind[1])
    ax_spd = fig.add_subplot(gs_wind[2], sharex=ax_r0)
    ax_dir = fig.add_subplot(gs_wind[3], sharex=ax_r0)

    # psf hexbin
    ax_hexPsf = hex_psf(fig, ax_hexPsf, thx, thy, sigma)

    # shear hexbin
    ax_hexSh = hex_shear(fig, ax_hexSh, thx, thy, e1, e2)

    # whisker
    ax_whisker = plot_whisker(ax_whisker, thx, thy, e1, e2)

    # psf hist
    ax_hisPsf = plt_hist(fig, ax_hisPsf, sigma, "thistle", "PSF width", "Number of PSF")

    # e1 hist
    ax_hise1 = plt_hist(fig, ax_hise1, e1, "mediumpurple", "e1", "Number of PSF")

    # e2 hist
    ax_hise2 = plt_hist(fig, ax_hise2, e2, "cornflowerblue", "e2", "Number of PSF")

    # e1, e2 hist
    ax_his2d.hist2d(e1,e2,cmap="BuPu",bins=30)
    fig_config(fig, ax_his2d, "e1", "e2")

    # r0
    ax_r0 = plt_wind(fig, ax_r0, layers, r0w, xi, "orchid", "Altitude", "$C^2_n(h)$")

    # speed
    ax_spd = plt_wind(fig, ax_spd, layers, spd, xi, "royalblue", "Altitude", "Speed")

    # wind dir
    ax_dir = plt_wind(fig, ax_dir, layers, direc, xi, "palevioletred", "Altitude", "Direction")
    savepath = os.path.join(args.outdir, args.imageF)
    fig.savefig(savepath)

if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        script to plot wide field simulation results
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--outfile", type=str, default="out.pkl", help="input pickle file")
    parser.add_argument("--imageF", type=str, default="wfresult.png", help="result graphs file")
    parser.add_argument('--outdir', type=str, default='output')
    args = parser.parse_args()
    plot_results(args)
