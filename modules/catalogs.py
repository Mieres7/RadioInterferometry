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
        'LMC': {'RA': '05:23:35', 'Dec': '-69:45:22'},
        'TEST1': {'RA':  0, 'Dec': -30},
    }

INTERFEROMETER_BANDS = {
    'VLA' : {
        "L": (1.0e9, 2.0e9),
        "S": (2.0e9, 4.0e9),
        "C": (4.0e9, 8.0e9),
        "X": (8.0e9, 12.0e9),
        "KU": (12.0e9, 18.0e9),
        "K": (18.0e9, 26.5e9),
        "KA": (26.5e9, 40.0e9),
        "Q": (40.0e9, 50.0e9),
    },
    'ALMA' : {
        "BAND 1":  (35.0e9, 50.0e9),
        "BAND 2":  (67.0e9, 116.0e9),
        "BAND 3":  (84.0e9, 116.0e9),
        "BAND 4":  (125.0e9, 163.0e9),
        "BAND 5":  (163.0e9, 211.0e9),
        "BAND 6":  (211.0e9, 275.0e9),
        "BAND 7":  (275.0e9, 373.0e9),
        "BAND 8":  (385.0e9, 500.0e9),
        "BAND 9":  (602.0e9, 720.0e9),
        "BAND 10": (787.0e9, 950.0e9),
    },
    'SKA_LOW':{
        "BAND": (50e6, 350e6)
    }
}


def select_frequencies(band_name, interferometer_band, num_frequencies=4):
    """
    Selects a band and generates a specified number of frequencies from its range.
    """
    band = band_name.upper() # Make it case-insensitive
    if band not in INTERFEROMETER_BANDS[interferometer_band]:
        raise ValueError(f"Band '{band}' not recognized. Available bands: {list(INTERFEROMETER_BANDS[interferometer_band].keys())}")

    min_freq, max_freq = INTERFEROMETER_BANDS[interferometer_band][band]
    
    # Generate evenly spaced frequencies within the selected band
    frequencies = np.linspace(min_freq, max_freq, num_frequencies)
    
    channel_bandwidth = frequencies[1] - frequencies[0]

    return frequencies, channel_bandwidth