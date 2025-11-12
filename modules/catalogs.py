import numpy as np

SOURCES_CATALOG = {
        'Sirius': {'RA': '06:45:09', 'Dec': '-16:42:58'},
        'Canopus': {'RA': '06:23:57', 'Dec': '-52:41:44'},
        'Centauri': {'RA': '14:39:36', 'Dec': '-60:50:02'},
        'Betelgeuse': {'RA': '05:55:10', 'Dec': '+07:24:25'},
        'Rigel': {'RA': '05:14:32', 'Dec': '-08:12:06'},
        'M31': {'RA': '00:42:44.3', 'Dec': '+41:16:09'},
        'M42': {'RA': '05:35:17.3', 'Dec': '-05:23:28'},
        '47_Tuc': {'RA': '00:24:05', 'Dec': '-72:04:52'},
        'LMC': {'RA': '05:23:35', 'Dec': '-69:45:22'}
    }

VLA_BANDS = {
    "L": (1.0e9, 2.0e9),
    "S": (2.0e9, 4.0e9),
    "C": (4.0e9, 8.0e9),
    "X": (8.0e9, 12.0e9),
    "Ku": (12.0e9, 18.0e9),
    "K": (18.0e9, 26.5e9),
    "Ka": (26.5e9, 40.0e9),
    "Q": (40.0e9, 50.0e9),
}

def select_vla_frequencies(band_name, num_frequencies=4):
    """
    Selects a VLA band and generates a specified number of frequencies from its range.
    """
    band = band_name.upper() # Make it case-insensitive
    if band not in VLA_BANDS:
        raise ValueError(f"Band '{band}' not recognized. Available bands: {list(VLA_BANDS.keys())}")

    min_freq, max_freq = VLA_BANDS[band]
    
    # Generate evenly spaced frequencies within the selected band
    frequencies = np.linspace(min_freq, max_freq, num_frequencies)
    
    return frequencies