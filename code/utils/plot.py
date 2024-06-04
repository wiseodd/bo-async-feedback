import matplotlib as mpl
import matplotlib.font_manager as font_manager

cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + "/fonts/ttf/cmr10.ttf")
FONT_NAME_CM = cmfont.get_name()
FONT_NAME_TNR = "Times New Roman"
FONT_NAME_AVENIR = "Avenir Next Condensed"

PAPER_FORMATS = {
    "icml": {
        "text_width": 6.00117,
        "col_width": 3.25063,
        "text_height": 8.50166,
        "font_name": FONT_NAME_TNR,
    },
    "neurips": {
        "text_width": 5.50107,
        "col_width": 5.50107,
        "text_height": 9.00177,
        "font_name": FONT_NAME_TNR,
    },
    "iclr": {
        "text_width": 5.50107,
        "col_width": 5.50107,
        "text_height": 9.00177,
        "font_name": FONT_NAME_TNR,
    },
    "jmlr": {
        "text_width": 6.00117,
        "col_width": 6.00117,
        "text_height": 8.50166,
        "font_name": FONT_NAME_CM,
    },
    "poster-landscape": {
        "text_width": 6.00117,
        "col_width": 6.00117,
        "text_height": 8.50166,
        "font_name": FONT_NAME_AVENIR,
    },
    "poster-portrait": {
        "text_width": 6.00117,
        "col_width": 6.00117,
        "text_height": 8.50166,
        "font_name": FONT_NAME_AVENIR,
    },
}

FOOTNOTE_SIZE = 8
SCRIPT_SIZE = 7


def get_mpl_rcParams(width_percent, height_percent, single_col=False, layout="neurips"):
    if layout not in PAPER_FORMATS.keys():
        raise ValueError(f"Layout must be in {list(PAPER_FORMATS.keys())}.")

    if layout not in ["icml"] and not single_col:
        raise ValueError("Double-column is only supported for ICML.")

    params = {
        "text.usetex": False,
        "font.size": SCRIPT_SIZE,
        "font.family": "serif",
        "font.serif": PAPER_FORMATS[layout]["font_name"],
        "mathtext.fontset": "cm",
        "lines.linewidth": 1.25,
        "axes.linewidth": 0.5,
        "axes.titlesize": FOOTNOTE_SIZE,
        "axes.labelsize": SCRIPT_SIZE,
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.facecolor": "white",
        "legend.edgecolor": "black",
        "legend.fontsize": 6,
        "legend.handlelength": 2,
        "xtick.major.size": 1.5,
        "ytick.major.size": 1.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    }

    w = width_percent * (
        PAPER_FORMATS[layout]["col_width"]
        if single_col
        else PAPER_FORMATS[layout]["text_width"]
    )
    h = height_percent * PAPER_FORMATS[layout]["text_height"]

    return params, w, h
