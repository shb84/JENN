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


MARKERS = {
    'o': 'Circle',
    'x': 'X',
    '.': 'Point',
    '+': 'Plus',
    ',': 'Pixel',
    'o': 'Circle',
    'v': 'Triangle Down',
    '^': 'Triangle Up',
    '<': 'Triangle Left',
    '>': 'Triangle Right',
    '1': 'Tri Down',
    '2': 'Tri Up',
    '3': 'Tri Left',
    '4': 'Tri Right',
    '8': 'Octagon',
    's': 'Square',
    'p': 'Pentagon',
    'P': 'Plus (filled)',
    '*': 'Star',
    'h': 'Hexagon 1',
    'H': 'Hexagon 2',
    '+': 'Plus',
    'X': 'X (filled)',
    'D': 'Diamond',
    'd': 'Thin Diamond',
    '|': 'Vline',
    '_': 'Hline',
    None: 'No marker', # Using None or ' ' or '' or 'none' for no marker
}
