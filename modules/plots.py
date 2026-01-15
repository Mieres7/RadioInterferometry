"""
Generación de Gráficos
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
from typing import List, Tuple, Any, Dict, Optional

from .interferometry import uvw_to_lambda, get_dirty_image
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
    color_by=None,
    # is_lambda=True # uvw is already in lambda units
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
    Funciona tanto con arrays de NumPy como de CuPy.
    """
    image_real = get_dirty_image(VG)

    # Graficar
    plt.figure(figsize=(8, 8))

    extent = None
    xlabel = "Pixeles"

    if pixel_size_arcsec is not None:
        N = VG.shape[0]
        fov = (N * pixel_size_arcsec) / 2
        extent = [-fov, fov, -fov, fov]
        xlabel = "Arcsec"

    plt.imshow(image_real, 
               origin='lower',
               cmap='inferno',
               extent=extent)

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

    # Configuración de scale
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

def plot_sources(l_src, m_src, sources):
    import numpy as np
    plt.figure(figsize=(8, 6))
    # El tamaño del punto (s) depende del flujo de la fuente para visualizarlas mejor
    fluxes = [src['S0'] for src in sources]
    scatter = plt.scatter(l_src, m_src, s=np.array(fluxes)*50, c=fluxes, cmap='viridis', alpha=0.8)
    
    plt.colorbar(scatter, label='Flujo (S0) [Jy]')
    plt.title('Distribución de Fuentes Puntuales (Cosenos Directores)')
    plt.xlabel('l (radianes)')
    plt.ylabel('m (radianes)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot(
    data: List[Dict[str, Any]], 
    grid_config: Tuple[int, int],
    scale: Optional[float] = None,
    fig_title: Optional[str] = None
):
    """
    Plots a grid of images OR 1D-curves based on a list of dictionaries.
    
    Args:
        data: List of dicts containing 'image', 'title', 'xlabel', 'ylabel', 'cmap'.
        grid_config: Tuple (rows, cols).
        scale: Multiplier for figure size. Calculated automatically if None.
    """
    rows, cols = grid_config
    total_subplots = rows * cols
    N = len(data)

    # Dynamic scaling logic
    if scale is None:
        if total_subplots == 1:
            scale = 8.0
        elif total_subplots <= 4:
            scale = 6.0
        elif total_subplots <= 12:
            scale = 4.0
        elif total_subplots <= 30:
            scale = 3.0
        else:
            scale = 2.0
    
    fig_width = scale * cols
    fig_height = scale * rows
    
    print(f"Generating plot with scale: {scale} (Figure size: {fig_width:.1f}x{fig_height:.1f} inches)")

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    if fig_title:
        fig.suptitle(fig_title, fontsize=scale * 3.5, y=0.98 if rows > 1 else 1.05)

    # Unify axes list
    if rows * cols == 1:
        axes_list = [axes]
    elif rows == 1 or cols == 1:
        axes_list = axes.tolist()
    else:
        axes_list = axes.flatten().tolist()

    for i in range(min(N, total_subplots)):
        ax = axes_list[i]
        item = data[i]
        
        img = item.get('image')

        # CuPy → NumPy
        if hasattr(img, 'get'):
            img = img.get()
        # PyTorch → NumPy
        elif hasattr(img, 'cpu'):
            img = img.cpu().numpy()

        title = item.get('title', '')
        xlabel = item.get('xlabel', '')
        ylabel = item.get('ylabel', '')
        cmap   = item.get('cmap', None)

        font_scale = max(8, scale * 2.5)

        # -----------------------------------
        # 🔍 Detectar si es curva 1D
        # -----------------------------------
        if np.asarray(img).ndim == 1:
            ax.plot(img, linewidth=2)
            ax.set_title(title, fontsize=font_scale)
            if xlabel: ax.set_xlabel(xlabel, fontsize=font_scale * 0.8)
            if ylabel: ax.set_ylabel(ylabel, fontsize=font_scale * 0.8)
            ax.grid(True)
            continue

        # -----------------------------------
        # 🔍 Detectar si es imagen RGB
        # -----------------------------------
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            ax.imshow(img, origin='lower')
            ax.set_title(title, fontsize=font_scale)
            if xlabel: ax.set_xlabel(xlabel, fontsize=font_scale * 0.8)
            if ylabel: ax.set_ylabel(ylabel, fontsize=font_scale * 0.8)
            continue

        # -----------------------------------
        # Caso normal: imagen 2D
        # -----------------------------------
        im = ax.imshow(img, cmap=cmap, origin='lower')
        ax.set_title(title, fontsize=font_scale)
        
        if xlabel: ax.set_xlabel(xlabel, fontsize=font_scale * 0.8)
        if ylabel: ax.set_ylabel(ylabel, fontsize=font_scale * 0.8)

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    # Ocultar subplots vacíos
    for j in range(N, total_subplots):
        axes_list[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_uv_coverage_2(uvw, bins=512):
    # Extraemos u y v (asegúrate de que estén en las unidades deseadas, ej. baselines)
    u = uvw[..., 0].flatten()
    v = uvw[..., 1].flatten()
    
    # El paper asume simetría conjugada completa
    u_full = np.concatenate([u, -u])
    v_full = np.concatenate([v, -v])
    
    plt.figure(figsize=(8, 7))
    
    # Histograma 2D con escala logarítmica para n_B
    # El rango debe coincidir con la extensión de tus baselines (ej. -250 a 250)
    h = plt.hist2d(u_full, v_full, bins=bins, 
                   cmap='viridis', # O 'YlGnBu_r' para un look similar
                   norm=colors.LogNorm(vmin=1, vmax=1e6)) # Rango 10^0 a 10^6 como el paper
    
    plt.colorbar(label='$n_B$')
    plt.xlabel('u')
    plt.ylabel('v')
    plt.title('Baseline Distribution (SKA-Low Replica)')
    plt.axis('equal')
    plt.show()


def plot_time_frequency(
    V,
    frequencies,
    eps=1e-6,
    figsize=(10, 6),
    title_suffix=""
):
    """
    Waterfall plot (time vs frequency) for amplitude and phase
    """

    frequencies = np.asarray(frequencies)

    # Average over baselines
    V_tf = np.mean(V, axis=0)  # (time, freq)

    # --- Amplitude ---
    amp = np.abs(V_tf)
    amp_log = np.log10(amp / np.median(amp) + eps)

    vmin = np.percentile(amp_log, 5)
    vmax = np.percentile(amp_log, 99.5)

    # --- Phase ---
    phase = np.angle(V_tf)

    extent = [
        frequencies.min() / 1e6,
        frequencies.max() / 1e6,
        0,
        amp_log.shape[0]
    ]

    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)

    im0 = axs[0].imshow(
        amp_log,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=extent
    )
    axs[0].set_ylabel("Time Integrations")
    axs[0].set_title(f"Amplitude (log scale) {title_suffix}")
    plt.colorbar(im0, ax=axs[0], label="log₁₀(|V| / median)")

    im1 = axs[1].imshow(
        phase,
        aspect="auto",
        origin="lower",
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        extent=extent
    )
    axs[1].set_xlabel("Frequency [MHz]")
    axs[1].set_ylabel("Time Integrations")
    axs[1].set_title(f"Phase {title_suffix}")
    plt.colorbar(im1, ax=axs[1], label="Phase [rad]")

    plt.tight_layout()
    plt.show()


def plot_rfi_spectrum(
    V,
    frequencies,
    eps=1e-12,
    log_mode="log10",
    normalize=True,
    figsize=(10, 4),
    title=""
):
    """
    Plot scalar-averaged cross-power spectrum for RFI inspection.

    Parameters
    ----------
    V : complex ndarray (N_baselines, N_times, N_freqs)
        Visibilities.
    frequencies : ndarray (N_freqs,)
        Frequencies in Hz.
    eps : float
        Small value to avoid log(0).
    log_mode : {'log10', 'dB'}
        Log scaling.
    normalize : bool
        Normalize by median amplitude.
    """

    frequencies = np.asarray(frequencies)

    # --- scalar average ---
    amp = np.abs(V)
    amp_mean = amp.mean(axis=(0, 1))  # average over baselines & time

    if normalize:
        amp_mean = amp_mean / np.median(amp_mean)

    # --- log scaling ---
    if log_mode == "log10":
        y = np.log10(amp_mean + eps)
        ylabel = r"log$_{10}$(|V|)"
    elif log_mode == "dB":
        y = 10 * np.log10(amp_mean + eps)
        ylabel = r"10 log$_{10}$(|V|)"
    else:
        raise ValueError("log_mode must be 'log10' or 'dB'")

    # --- plot ---
    plt.figure(figsize=figsize)
    plt.plot(frequencies / 1e6, y, lw=1.0, color="black")

    plt.xlabel("Frequency [MHz]")
    plt.ylabel(ylabel)
    plt.title(title or "Scalar-averaged cross-power spectrum")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()