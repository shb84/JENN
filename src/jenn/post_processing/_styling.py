# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.
 
LINE_STYLES = {
    "solid": "solid",  # Same as (0, ()) or '-'
    "dotted": "dotted",  # Same as (0, (1, 1)) or ':'
    "dashdot": "dashdot",  # Same as '-.'
    "dashed": "dashed",  # Same as '--'
    "loosely dotted": (0, (1, 10)),
    # "dotted": (0, (1, 1)),
    "densely dotted": (0, (1, 1)),
    "long dash with offset": (5, (10, 3)),
    "loosely dashed": (0, (5, 10)),
    # "dashed": (0, (5, 5)),
    "densely dashed": (0, (5, 1)),
    "loosely dashdotted": (0, (3, 10, 1, 10)),
    "dashdotted": (0, (3, 5, 1, 5)),
    "densely dashdotted": (0, (3, 1, 1, 1)),
    "dashdotdotted": (0, (3, 5, 1, 5, 1, 5)),
    "loosely dashdotdotted": (0, (3, 10, 1, 10, 1, 10)),
    "densely dashdotdotted": (0, (3, 1, 1, 1, 1, 1)),
}