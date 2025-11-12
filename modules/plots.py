"""
Generación de Gráficos
"""

import numpy as np
import matplotlib.pyplot as plt
import math

from .interferometry import uvw_to_lambda, to_fourier
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_uv_coverage(
    uvw,
    unit="m",
    freq_hz=100e9,
    title=None,
    show_symmetry=True,
    show_center=True,
    center_color="white",
    center_size=50,
    color="deepskyblue",
    symmetry_color=None,
    cmap=None,
    color_by=None
):
    """
    Plot uvw coverage
    """
    # --- Selección de unidades ---
    if unit == "m":
        scale = 1.0
        label = "[m]"
    elif unit == "km":
        scale = 1e-3
        label = "[km]"
    elif unit.lower() in ["λ", "lambda", "wavelength"]:
        uvw, lam = uvw_to_lambda(uvw, freq_hz)
        scale = 1.0
        label = "[λ]"
    else:
        raise ValueError("Unidad no reconocida. Usa 'm', 'km' o 'lambda'.")

    u = uvw[..., 0] * scale
    v = uvw[..., 1] * scale

    if symmetry_color is None:
        if isinstance(color, str):
            symmetry_color = "royalblue" if color == "deepskyblue" else color
        else:
            symmetry_color = color

    plt.figure(figsize=(6, 6))

    if color_by is not None:
        sc = plt.scatter(u.flatten(), v.flatten(), c=color_by.flatten(),
                         s=2, cmap=cmap or "viridis", label="Baselines")
        plt.colorbar(sc, label="Color variable")
    else:
        plt.plot(u.flatten(), v.flatten(), ".", markersize=1, color=color, label="Baselines")

    if show_symmetry:
        plt.plot(-u.flatten(), -v.flatten(), ".", markersize=1,
                 alpha=0.4, color=symmetry_color, label="Simetría conjugada")

    # --- Punto central ---
    if show_center:
        plt.scatter(0, 0, color=center_color, s=center_size, zorder=5)

    plt.xlabel(f"u {label}")
    plt.ylabel(f"v {label}")
    plt.axis("equal")
    plt.grid(alpha=0.2)
    plt.legend(markerscale=5, fontsize=8, loc="upper right")

    if title:
        plt.title(title)
    else:
        plt.title(f"Cobertura UV ({unit})")

    plt.show()


def plot_antennas(enu_coords, labels=True, title="Configuración de Antenas", unit="m"):
    """
    Muestra la configuración de antenas a partir de sus coordenadas ENU.

    Parámetros
    ----------
    enu_coords : ndarray (N,3)
        Arreglo con las coordenadas [E, N, U] de cada antena (en metros).
    labels : bool
        Si True, muestra el índice de cada antena en el gráfico.
    title : str
        Título del gráfico.
    unit : {"m", "km"}
        Unidad en la que se mostrarán las coordenadas en el gráfico.
        Siempre se asume que enu_coords está en metros internamente.
    """

    enu_coords = np.array(enu_coords)

    # --- Conversión de unidades ---
    if unit == "m":
        scale = 1
        unit_label = "m"
    elif unit == "km":
        scale = 1e-3
        unit_label = "km"
    else:
        raise ValueError(f"Unidad desconocida: {unit}. Usa 'm' o 'km'.")

    E, N, U = enu_coords[:, 0] * scale, enu_coords[:, 1] * scale, enu_coords[:, 2] * scale

    # --- Gráfico ---
    fig = plt.figure(figsize=(12, 6))

    # Vista en planta (E-N)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(E, N, c='black', marker='o')
    if labels:
        for i, (e, n) in enumerate(zip(E, N)):
            ax1.text(e, n, str(i), fontsize=9, ha='right')
    ax1.set_xlabel(f"East [{unit_label}]")
    ax1.set_ylabel(f"North [{unit_label}]")
    ax1.set_title("Vista en planta (E-N)")
    ax1.grid(True)
    ax1.axis('equal')

    # Vista en perfil (N-U)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(N, U, c='darkblue', marker='o')
    if labels:
        for i, (n, u) in enumerate(zip(N, U)):
            ax2.text(n, u, str(i), fontsize=9, ha='right')
    ax2.set_xlabel(f"North [{unit_label}]")
    ax2.set_ylabel(f"Up [{unit_label}]")
    ax2.set_title("Vista en perfil (N-U)")
    ax2.grid(True)
    ax2.axis('equal')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()



def plot_all_dirty_images(VG, cell_size_arcsec):
    """
    Calcula y grafica la "dirty image" para cada canal de frecuencia en el cubo VG.
    """
    N, _, num_channels = VG.shape
    
    # Configura una grilla de subplots para mostrar todas las imágenes
    cols = math.ceil(math.sqrt(num_channels))
    rows = math.ceil(num_channels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), squeeze=False)
    fig.suptitle('Dirty Images por Canal de Frecuencia', fontsize=16)

    # Calcula el campo de visión para etiquetar los ejes correctamente
    image_fov_arcsec = N * cell_size_arcsec
    extent = [-image_fov_arcsec / 2, image_fov_arcsec / 2, -image_fov_arcsec / 2, image_fov_arcsec / 2]
    extent=[-1,1,-1,1]

    for i in range(num_channels):
        ax = axes[i // cols, i % cols]
        
        # Selecciona el canal y calcula la imagen
        image = to_fourier(VG[..., i])
        intensity = np.abs(image)
        
        im = ax.imshow(intensity, origin='lower', cmap='inferno', 
                       extent=extent)
        
        ax.set_title(f'Canal {i}')
        ax.set_xlabel('Offset RA (arcsec)')
        ax.set_ylabel('Offset Dec (arcsec)')

        # 1. Crea un divisor para el eje actual
        divider = make_axes_locatable(ax)
        # 2. Añade un nuevo eje a la derecha, del 5% del ancho de la imagen y con un poco de padding
        cax = divider.append_axes("right", size="5%", pad=0.1)
        # 3. Dibuja el colorbar en ese nuevo eje específico
        fig.colorbar(im, cax=cax, label='Intensidad')

    # Oculta los ejes de los subplots que no se usen
    for i in range(num_channels, rows * cols):
        axes[i // cols, i % cols].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()




def calculate_and_plot_psf(WG, cell_size_arcsec):
    """
    Calcula y grafica la PSF ("dirty beam") para cada canal de frecuencia.
    """
    N, _, num_channels = WG.shape

    cols = math.ceil(math.sqrt(num_channels))
    rows = math.ceil(num_channels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
    fig.suptitle('PSF ("Dirty Beam") por Canal de Frecuencia', fontsize=16)

    image_fov_arcsec = N * cell_size_arcsec
    extent = [-image_fov_arcsec / 2, image_fov_arcsec / 2, -image_fov_arcsec / 2, image_fov_arcsec / 2]
    extent=[-1,1,-1,1]

    for i in range(num_channels):
        ax = axes[i // cols, i % cols]
        
        # Para la PSF, usamos la grilla de pesos WG
        # Normalizamos la grilla de pesos para que el pico central de la PSF sea 1
        wg_channel = WG[..., i]
        if np.max(wg_channel) > 0:
            wg_channel = wg_channel / np.max(wg_channel)

        psf = to_fourier(wg_channel)
        intensity = np.abs(psf)
        
        im = ax.imshow(intensity, origin='lower', cmap='viridis', 
                       extent=extent)
        
        ax.set_title(f'PSF Canal {i}')
        ax.set_xlabel('Offset RA (arcsec)')
        ax.set_ylabel('Offset Dec (arcsec)')

        
        # 1. Crea un divisor para el eje actual
        divider = make_axes_locatable(ax)
        # 2. Añade un nuevo eje a la derecha, del 5% del ancho de la imagen y con un poco de padding
        cax = divider.append_axes("right", size="5%", pad=0.1)
        # 3. Dibuja el colorbar en ese nuevo eje específico
        fig.colorbar(im, cax=cax, label='Respuesta Normalizada')

    for i in range(num_channels, rows * cols):
        axes[i // cols, i % cols].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
