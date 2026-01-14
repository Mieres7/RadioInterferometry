import numpy as np
import pytest as pt

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

# Test 1: Visibilidades sintéticas no grideadas con ruido gaussiano
def test_rms_ruido_gaussiano():

    np.random.seed(42)
    N = 10000
    
    # Ruido gaussiano
    ruido_real = np.random.randn(N)
    ruido_imag = np.random.randn(N)
    
    visibilidades_ruidosas = ruido_real + 1j * ruido_imag
    
    rms_calculado = calcular_ruido_rms(visibilidades_ruidosas)
    rms_esperado = 1.0
    
    assert np.isclose(rms_calculado, rms_esperado, atol=0.05), \
        f"El RMS calculado ({rms_calculado}) no es cercano al esperado (1.0) para ruido gaussiano."
    
# Test 2: Visibilidades con un solo valor
def test_rms_un_solo_valor():

    visibilidades_single = np.array([10.5 + 5.2j])
    rms_calculado = calcular_ruido_rms(visibilidades_single)
    
    assert rms_calculado == 0.0, \
        f"El RMS de un solo valor debería ser 0.0, pero se obtuvo {rms_calculado}."
    
# Test 3: Visibilidades vacías
def test_rms_array_vacio():

    visibilidades_vacias = np.array([])
    
    with pt.raises(ValueError, match="no puede estar vacío"):
        calcular_ruido_rms(visibilidades_vacias)