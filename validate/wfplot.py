import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
import treecorr
import seaborn as sns

def fig_config(fig, ax, xlabel, ylabel):
    """
    setting labels for plots
    """
    ax.set_xlabel(xlabel, size=15, labelpad=3)
    ax.set_ylabel(ylabel, size=15)
    ax.tick_params(labelsize=15)
    # ax.set_title(title, size=15, pad=-50)


def make_grid(fig, gs):
    """
    making grids for the image
    """
    gs = fig.add_gridspec(nrows=3, ncols=3, width_ratios=[1.5, 2, 2], height_ratios=[2, 2, 1], hspace=0.5, wspace=0.5)
    # make grid
    gs_hexbin = gs[0, 1:].subgridspec(1, 2, wspace=0.5)
    gs_new = gs[1, 1:].subgridspec(1, 2, wspace=0.5)
    gs_hist = gs[2, 1:].subgridspec(1, 3, wspace=0.5)
    gs_wind = gs[:, 0].subgridspec(5, 1, hspace=0.5)
    return gs_hexbin, gs_new, gs_hist, gs_wind


def hex_psf(fig, ax_hexPsf, thx, thy, sigma, args):
    """
    2D hexbin plot of psf position with psf size as weight
    """
    color = sns.diverging_palette(150, 260, as_cmap=True)
    im0 = ax_hexPsf.hexbin(thx, thy, gridsize=30, C=(sigma-np.mean(sigma))*0.2, cmap=color, alpha=0.8)
    fig_config(fig, ax_hexPsf, "$\Theta_x$ [degree]", "$\Theta_y$ [degree]")
    ax_hexPsf.margins(0.1, 0.1)
    divider0 = make_axes_locatable(ax_hexPsf)
    cax0 = divider0.append_axes('top', size='5%', pad=0.1)
    cbr0 = plt.colorbar(im0, cax=cax0, orientation='horizontal')
    cbr0.set_label("$\delta \sigma$ [arcsec]", size=15)
    cbr0.formatter.set_powerlimits((-3, 3))
    cax0.xaxis.set_ticks_position("top")
    cax0.xaxis.set_label_position("top")
    cax0.tick_params(labelsize=15)
    return ax_hexPsf


def hex_shear(fig, ax_hexSh, thx, thy, e1, e2, args):
    """
    2D hexbin plot of psf position with shear magnitude as weight
    """
    e = np.hypot(e1,e2)
    im1 = ax_hexSh.hexbin(thx, thy, gridsize=30, C=e, cmap="BuPu")
    fig_config(fig, ax_hexSh, "$\Theta_x$ [degree]", "$\Theta_y$ [degree]")
    ax_hexSh.margins(0.1, 0.1)
    divider1 = make_axes_locatable(ax_hexSh)
    cax1 = divider1.append_axes('top', size='5%', pad=0.1)
    cbr1 = plt.colorbar(im1, cax=cax1, orientation='horizontal')
    cbr1.set_label("|e| ", size=15)
    cbr1.formatter.set_powerlimits((-3, 3))
    cax1.xaxis.set_ticks_position("top")
    cax1.xaxis.set_label_position("top")
    cax1.tick_params(labelsize=15)
    return ax_hexSh


def plot_whisker(ax, thx, thy, e1, e2, args):
    """
    2D vector field plot of shear
    """
    e = np.hypot(e1, e2)
    beta = 0.5 * np.arctan2(e2, e1)
    dx = e * np.cos(beta)
    dy = e * np.sin(beta)
    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.9, 1.9)
    ax.set_xlabel('[degree]', fontsize=15)
    ax.set_ylabel("[degree]", fontsize=15)
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
    cbar = plt.colorbar(q, cax=ax_cb, orientation='horizontal')
    cbar.set_label(label='|e| ',fontsize=15)
    cbar.formatter.set_powerlimits((-3, 3))
    ax_cb.xaxis.set_ticks_position("top")
    ax_cb.xaxis.set_label_position("top")
    ax_cb.tick_params(labelsize=13)
    ax.tick_params(labelsize=13)
    return ax


def plt_corr(fig, ax, thx, thy, para, xlabel, ylabel, cbarlabel, axlabel, args):
    cat = treecorr.Catalog(x=thx, y=thy, k=para, x_units='degree', y_units='degree')
    kk = treecorr.KKCorrelation(min_sep=0, max_sep=0.5,bin_type="TwoD", nbins=20, sep_units='degree')
    kk.process(cat)
    xim = kk.xi
    #color = sns.diverging_palette(260, 150, as_cmap=True)
    color=sns.light_palette("seagreen", as_cmap=True)
    im = ax.imshow(xim, cmap=color, origin="lower", alpha=0.8)
    fig_config(fig, ax, xlabel, ylabel)
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes("right", size="4%", pad="4%")
    cbar = plt.colorbar(im, cax=ax_cb, orientation='vertical')
    cbar.set_label(label=cbarlabel,fontsize=15)
    cbar.formatter.set_powerlimits((0, 0))
    ax_cb.xaxis.set_ticks_position("bottom")
    ax_cb.xaxis.set_label_position("top")
    ax_cb.tick_params(labelsize=15)
    ax.set_xticks(np.linspace(0,19.5,3),np.linspace(-0.5,0.5,3))
    ax.set_yticks(np.linspace(0,19.5,3),np.linspace(-0.5,0.5,3))
    ax.set_title(axlabel, fontsize=15)
    ax.tick_params(labelsize=15)
    return ax


def plt_wind(fig, ax, layers, para, x, clr, xlabel, ylabel, args):
    """
    scatter plot with interpolated lines
    """
    f = interp1d(layers, para, kind="cubic")
    ax.scatter(layers, para, color=clr)
    if args.usePsfws:
        ax.plot(x, f(x), color=clr)
    fig_config(fig, ax, xlabel, ylabel)
    return ax

def his_2d(ax, e1, e2):
    ax.hist2d(e1,e2,cmap="BuPu",bins=50)
    ax.axhline(linewidth=1, color='k')
    ax.axvline(linewidth=1, color='k')
    ax.text(x=0.7,y=0.7,s="$\\rho$ = %.3f" %(np.corrcoef(e1,e2)[0,1]),transform=ax.transAxes, size=17)
    ax.set_xlabel("e1", loc="right",size=15)
    ax.set_ylabel("e2", loc="top",size=15, labelpad=-10)
    ax.tick_params(labelsize=15)
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


    fig = plt.figure(figsize=(20,18))
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
    ax_hexPsf = hex_psf(fig, ax_hexPsf, thx, thy, sigma, args)

    # shear hexbin
    ax_hexSh = hex_shear(fig, ax_hexSh, thx, thy, e1, e2,args)

    # whisker
    ax_whisker = plot_whisker(ax_whisker, thx, thy, e1, e2, args)

    # sigma corr
<<<<<<< HEAD:runsims/wfplot.py
    ax_hisPsf = plt_corr(fig, ax_hisPsf, thx, thy, sigma, "$\Delta \Theta$ x[degree]", "$\Delta \Theta$ y[degree]", "$cov_{i,j}$", "$\sigma$ corr")


    # e1 corr
    ax_hise1 = plt_corr(fig, ax_hise1, thx, thy, e1, "$\Delta \Theta$ x[degree]", "", "$cov_{i,j}$", "e1 corr")

    # e2 corr
    ax_hise2 = plt_corr(fig, ax_hise2, thx, thy, e2, "$\Delta \Theta$ x[degree]", "", "$cov_{i,j}$", "e2 corr")
=======
    ax_hisPsf = plt_corr(fig, ax_hisPsf, thx, thy, sigma, "$\Delta \Theta$ x[degree]", "$\Delta \Theta$ y[degree]", "$cov_{i,j}$", "$\\xi_{\sigma}$ ", args)


    # e1 corr
    ax_hise1 = plt_corr(fig, ax_hise1, thx, thy, e1, "$\Delta \Theta$ x[degree]", "", "$cov_{i,j}$", "$\\xi_{e1}$ ", args)

    # e2 corr
    ax_hise2 = plt_corr(fig, ax_hise2, thx, thy, e2, "$\Delta \Theta$ x[degree]", "", "$cov_{i,j}$", "$\\xi_{e2}$ ", args)
    
    # e1, e2 hist
    ax_his2d = his_2d(ax_his2d, e1, e2)

    # r0
    if args.usePsfws:
        ax_r0 = plt_wind(fig, ax_r0, layers, r0w, xi, "purple", "Altitude [km]", "$C^2_n(h)$ [$m^{-2/3}$]", args)
    elif args.useRand:
        ax_r0 = plt_wind(fig, ax_r0, layers, r0w, xi, "purple", "Altitude [km]", "J(h) [$m^{1/3}$]", args)

    # speed
    ax_spd = plt_wind(fig, ax_spd, layers, spd, xi, "royalblue", "Altitude [km]", "Speed [m/s]", args)

    # wind dir
    ax_dir = plt_wind(fig, ax_dir, layers, direc, xi, "mediumseagreen", "Altitude [km]", "Direction [degree]", args)
    ax_dir.set_ylim([0,360])
    ax_dir.set_yticks([0, 90, 180, 270, 360])

    savepath = os.path.join(args.imageDir, args.imageF)
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
    parser.add_argument("--imageDir", type=str)
    parser.add_argument("--usePsfws", action='store_true')
    parser.add_argument('--useRand', action='store_true')
    parser.add_argument("--kind", type=str)
    args = parser.parse_args()
    plot_results(args)
