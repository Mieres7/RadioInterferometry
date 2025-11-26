"""
Generación de Gráficos
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import math

from .interferometry import uvw_to_lambda, to_fourier, to_image
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



def plot_dirty_image(VG, pixel_size_arcsec=None, title="Dirty Image"):
    """
    Calcula y grafica la Dirty Image a partir de la grilla de visibilidades.
    """
    # 1. Detectar backend (GPU/CPU)
    xp = cp.get_array_module(VG)
    
    # 2. Transformada Inversa (IFFT)
    # Secuencia: ifftshift -> ifft2 -> fftshift para centrar la imagen
    image_complex = to_image(VG)

    # 3. Extraer parte Real y mover a CPU para graficar
    if xp == cp:
        image_real = cp.asnumpy(image_complex.real)
    else:
        image_real = image_complex.real

    # 4. Graficar
    plt.figure(figsize=(8, 8))
    
    extent = None
    xlabel = "Pixeles"
    if pixel_size_arcsec:
        N = VG.shape[0]
        fov = (N * pixel_size_arcsec) / 2
        extent = [-fov, fov, -fov, fov]
        xlabel = "Arcsec"

    plt.imshow(image_real, origin='lower', cmap='inferno', extent=extent)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(xlabel)
    plt.colorbar(label="Intensidad (Jy/beam)")
    plt.show()
    
    return image_real

def plot_psf(WG, pixel_size_arcsec=None, log_scale=False, title="PSF (Dirty Beam)"):
    """
    Calcula y grafica la PSF a partir de la grilla de pesos.
    """
    # 1. Detectar backend
    xp = cp.get_array_module(WG)
    
    # 2. Transformada Inversa de los pesos
    psf_complex = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(WG)))
    
    # 3. Magnitud absoluta y mover a CPU
    if xp == cp:
        psf_abs = cp.asnumpy(cp.abs(psf_complex))
    else:
        psf_abs = np.abs(psf_complex)

    # 4. Graficar
    plt.figure(figsize=(8, 8))
    
    extent = None
    xlabel = "Pixeles"
    if pixel_size_arcsec:
        N = WG.shape[0]
        fov = (N * pixel_size_arcsec) / 2
        extent = [-fov, fov, -fov, fov]
        xlabel = "Arcsec"

    # Configuración de escala
    norm = None
    if log_scale:
        from matplotlib.colors import LogNorm
        # Evitamos log(0) usando un vmin pequeño relativo al máximo
        norm = LogNorm(vmin=max(psf_abs.max()*1e-4, 1e-10), vmax=psf_abs.max())

    plt.imshow(psf_abs, origin='lower', cmap='viridis', extent=extent, norm=norm)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(xlabel)
    plt.colorbar(label="Respuesta Normalizada")
    plt.show()
    
    return psf_abs


def plot_time_vs_grid(df):
    """
    Grafica el tiempo de ejecución vs el tamaño de la grilla (N)
    replicando el estilo de la imagen de referencia.
    
    Parámetros:
    - df: DataFrame con las columnas 'Grid Size', 'CPU (s)', 'CuPy (s)', 'Numba (s)'
    """
    
    # Crear la figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Truco visual: Convertimos 'Grid Size' a string para que el eje X 
    # trate los valores como categorías equidistantes. 
    # Esto hace que la distancia entre 256->512 sea igual que 2048->4096,
    # tal como en tu imagen de referencia.
    x_labels = df["Grid Size"].astype(str)
    
    # 1. Plot CPU (Azul, círculo, línea punteada) -> 'bo--'
    # Usamos linewidth y markersize para que se vea robusto
    ax.plot(x_labels, df["CPU (s)"], 'bo--', label='CPU Numpy', 
            linewidth=1.5, markersize=6)

    # 2. Plot GPU CuPy (Verde, cuadrado, línea sólida) -> 'gs-'
    ax.plot(x_labels, df["CuPy (s)"], 'gs-', label='GPU CuPy', 
            linewidth=1.5, markersize=6)

    # 3. Plot GPU Numba (Rojo, triángulo, línea punto-guión) -> 'r^ -.'
    ax.plot(x_labels, df["Numba (s)"], 'r^-.', label='GPU Numba', 
            linewidth=1.5, markersize=6)

    # --- Estilizado ---
    
    # Escala Logarítmica en Y (Crucial para ver las diferencias)
    ax.set_yscale('log')
    
    # Etiquetas y Título
    ax.set_xlabel('Tamaño de Grilla (N)', fontsize=11)
    ax.set_ylabel('Tiempo de Ejecución (s) [Escala Log]', fontsize=11)
    ax.set_title('Comparación de Rendimiento: Tiempo vs Tamaño de Grilla', fontsize=13)
    
    # Grid (Rejilla)
    # 'which="both"' activa las líneas para la escala logarítmica (mayores y menores)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    
    # Leyenda
    ax.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.show()

def plot_speedups(df):
    plt.figure(figsize=(8,5))
    plt.plot(df["Grid Size"], df["Speedup Numba vs CPU"], marker="o", label="Numba vs CPU")
    plt.plot(df["Grid Size"], df["Speedup CuPy vs CPU"], marker="o", label="CuPy vs CPU")
    plt.plot(df["Grid Size"], df["Speedup Numba vs CuPy"], marker="o", label="Numba vs CuPy")

    plt.xlabel("Grid Size")
    plt.ylabel("Speedup (×)")
    plt.grid(True)
    plt.legend()
    plt.title("Comparación de Speedups")
    plt.show()