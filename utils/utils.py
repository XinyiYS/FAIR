import os
from contextlib import contextmanager

@contextmanager
def cwd(path):
    oldpwd=os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


# Plotting parameters

def set_up_plotting(use_latex=False):

    # from distutils.spawn import find_executable

    import seaborn as sns; sns.set_theme()
    import matplotlib.pyplot as plt

    LABEL_FONTSIZE = 22
    MARKER_SIZE = 10
    AXIS_FONTSIZE = 26
    TITLE_FONTSIZE= 26
    LINEWIDTH = 6

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('figure', titlesize=TITLE_FONTSIZE)     # fontsize of the axes title
    plt.rc('axes', titlesize=TITLE_FONTSIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=AXIS_FONTSIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=LABEL_FONTSIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=LABEL_FONTSIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LABEL_FONTSIZE)    # legend fontsize
    plt.rc('lines', markersize=MARKER_SIZE)  # fontsize of the figure title
    plt.rc('lines', linewidth=LINEWIDTH)  # fontsize of the figure title
    plt.rc('font', weight='bold') # set bold fonts


    # if use_latex and find_executable('latex'): 
    #     print("latex installed, using latex for matplotlib")
    #     plt.rcParams['text.usetex'] = True
    # else:
    #     plt.rcParams['text.usetex'] = False

    return plt
