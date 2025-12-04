import pytest
import numpy as np

def calcular_ruido_rms(visibilidades):
    visibilidades = np.array(visibilidades)

    if visibilidades.size == 0:
        raise ValueError("El conjunto de visibilidades no puede estar vacío.")

    parte_real = visibilidades.real
    parte_imag = visibilidades.imag
    rms_real = np.std(parte_real)
    rms_imag = np.std(parte_imag)

    rms_total = 1/2 * (rms_real + rms_imag)

    return rms_total

@pytest.mark.parametrize("n_elementos, nivel_ruido", [
    (100, 0.5),
    (100, 2.0),
    (1000, 1.0),
    (10000, 0.1), 
    (10000, 5.0)
])

def test_rms_parametrizado(n_elementos, nivel_ruido):
    """
    Prueba la función calcular_ruido_rms con diferentes tamaños de muestra
    y niveles de ruido.
    """

    ruido_real = np.random.normal(loc=0.0, scale=nivel_ruido, size=n_elementos)
    ruido_imag = np.random.normal(loc=0.0, scale=nivel_ruido, size=n_elementos)
    
    visibilidades = ruido_real + 1j * ruido_imag

    resultado_rms = calcular_ruido_rms(visibilidades)

    assert np.isclose(resultado_rms, nivel_ruido, rtol=0.2), \
        f"Error con N={n_elementos}, Sigma={nivel_ruido}. Se obtuvo {resultado_rms}"