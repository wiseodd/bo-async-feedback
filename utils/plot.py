from __future__ import annotations
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
        "footnote_size": 8,
        "script_size": 7,
        "linewidth": 1.25,
        "tick_size": 1,
        "tick_width": 1,
    },
    "neurips": {
        "text_width": 5.50107,
        "col_width": 5.50107,
        "text_height": 9.00177,
        "font_name": FONT_NAME_TNR,
        "footnote_size": 8,
        "script_size": 7,
        "linewidth": 1.25,
        "tick_size": 1,
        "tick_width": 1,
    },
    "iclr": {
        "text_width": 5.50107,
        "col_width": 5.50107,
        "text_height": 9.00177,
        "font_name": FONT_NAME_TNR,
        "footnote_size": 8,
        "script_size": 7,
        "linewidth": 1.25,
        "tick_size": 1,
        "tick_width": 1,
    },
    "jmlr": {
        "text_width": 6.00117,
        "col_width": 6.00117,
        "text_height": 8.50166,
        "font_name": FONT_NAME_CM,
        "footnote_size": 8,
        "script_size": 7,
        "linewidth": 1.25,
        "tick_size": 1,
        "tick_width": 1,
    },
    "poster-landscape": {
        "text_width": 6.00117,
        "col_width": 6.00117,
        "text_height": 8.50166,
        "font_name": FONT_NAME_AVENIR,
        "footnote_size": 30,
        "script_size": 23,
        "linewidth": 3,
        "tick_size": 4,
        "tick_width": 2,
    },
    "poster-portrait": {
        "text_width": 6.00117,
        "col_width": 6.00117,
        "text_height": 8.50166,
        "font_name": FONT_NAME_AVENIR,
        "footnote_size": 10,
        "script_size": 8,
        "linewidth": 1,
        "tick_size": 1,
        "tick_width": 1,
    },
}


def get_mpl_rcParams(width_percent, height_percent, single_col=False, layout="neurips"):
    if layout not in PAPER_FORMATS.keys():
        raise ValueError(f"Layout must be in {list(PAPER_FORMATS.keys())}.")

    if layout not in ["icml"] and single_col:
        raise ValueError("Double-column is only supported for ICML.")

    format = PAPER_FORMATS[layout]
    is_poster = "poster" in layout

    params = {
        "text.usetex": False,
        "font.size": format["footnote_size"],
        "font.family": "serif",
        "font.serif": format["font_name"],
        # "mathtext.fontset": "cm",
        "lines.linewidth": format["linewidth"],
        "axes.linewidth": 1,
        "axes.titlesize": format["footnote_size"],
        "axes.labelsize": format["script_size"],
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.facecolor": "white",
        "legend.edgecolor": "black",
        "legend.fontsize": format["script_size"],
        "legend.handlelength": 2,
        "xtick.major.size": format["tick_size"],
        "ytick.major.size": format["tick_size"],
        "xtick.major.width": format["tick_width"],
        "ytick.major.width": format["tick_width"],
    }

    w = width_percent * (format["col_width"] if single_col else format["text_width"])
    h = height_percent * format["text_height"]

    return params, w, h
