import matplotlib as mpl
import matplotlib.font_manager as font_manager


# ICML
# ------------------------
# TEXT_WIDTH = 6.00117
# COL_WIDTH = 3.25063
# TEXT_HEIGHT = 8.50166

# NeurIPS/ICLR
# ------------------------
TEXT_WIDTH = 5.50107
COL_WIDTH = 5.50107  # Here we have only single column
TEXT_HEIGHT = 9.00177

FOOTNOTE_SIZE = 8
SCRIPT_SIZE = 7

# cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
# FONT_NAME = cmfont.get_name()
FONT_NAME = 'Times New Roman'


def get_mpl_rcParams(width_percent, height_percent, single_col=False):
    params = {
        'text.usetex': False,
        'font.size': SCRIPT_SIZE,
        'font.family': 'serif',
        'font.serif': FONT_NAME,
        'mathtext.fontset': 'cm',
        'lines.linewidth': 1.25,
        'axes.linewidth': 0.5,
        'axes.titlesize': FOOTNOTE_SIZE,
        'axes.labelsize': SCRIPT_SIZE,
        'axes.unicode_minus': False,
        'axes.formatter.use_mathtext': True,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.facecolor': 'white',
        'legend.edgecolor': 'black',
        'legend.fontsize': 6,
        'legend.handlelength': 1,
        'xtick.major.size': 1.5,
        'ytick.major.size': 1.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
    }
    w = width_percent * (COL_WIDTH if single_col else TEXT_WIDTH)
    h = height_percent * TEXT_HEIGHT

    return params, w, h
