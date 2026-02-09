import numpy as np
import dask.array as da

def _adjoint_block(u_block, v_block, V_block, l, m):
    """
    Bloque local de A^H V
    u_block, v_block, V_block : (B,)
    l, m : (N, N)
    """
    phase = np.exp(
        2j * np.pi * (
            u_block[:, None, None] * l[None, :, :] +
            v_block[:, None, None] * m[None, :, :]
        )
    )
    return np.sum(V_block[:, None, None] * phase, axis=0)


def A_adjoint_dask(V, u, v, l, m, chunk_size=512):
    """
    A^H V usando Dask (seguro en memoria)
    """
    u_d = da.from_array(u, chunks=(chunk_size,))
    v_d = da.from_array(v, chunks=(chunk_size,))
    V_d = da.from_array(V, chunks=(chunk_size,))

    partials = da.map_blocks(
        _adjoint_block,
        u_d,
        v_d,
        V_d,
        dtype=np.complex128,
        new_axis=[1, 2],
        l=l,
        m=m,
    )

    return partials.sum(axis=0).compute()


def _forward_block(I, u_block, v_block, l, m):
    """
    Bloque local de A I
    """
    phase = np.exp(
        -2j * np.pi * (
            u_block[:, None, None] * l[None, :, :] +
            v_block[:, None, None] * m[None, :, :]
        )
    )
    return np.sum(I[None, :, :] * phase, axis=(1, 2))

def A_forward_dask(I, u, v, l, m, chunk_size=512):
    """
    A I usando Dask
    """
    u_d = da.from_array(u, chunks=(chunk_size,))
    v_d = da.from_array(v, chunks=(chunk_size,))

    V_model = da.map_blocks(
        _forward_block,
        I,
        u_d,
        v_d,
        dtype=np.complex128,
        l=l,
        m=m,
    )

    return V_model.compute()

def normal_operator_dask(I, u, v, l, m, chunk_size=512):
    return A_adjoint_dask(
        A_forward_dask(I, u, v, l, m, chunk_size),
        u, v, l, m, chunk_size
    )

def conjugate_gradient_imaging(
    Vobs,
    u,
    v,
    l,
    m,
    I0=None,
    tol=1e-6,
    maxiter=20,
    chunk_size=512,
    verbose=True,
):
    """
    Reconstrucción de imagen por Gradiente Conjugado usando Dask
    """

    N = l.shape[0]

    if I0 is None:
        I = np.zeros((N, N), dtype=np.complex128)
    else:
        I = I0.astype(np.complex128)

    # b = A^H V
    b = A_adjoint_dask(Vobs, u, v, l, m, chunk_size)

    # r0 = b - A^H A I
    r = b - normal_operator_dask(I, u, v, l, m, chunk_size)
    p = r.copy()

    rsold = np.vdot(r, r).real

    if verbose:
        print(f"CG iter 00 | residual = {np.sqrt(rsold):.3e}")

    for k in range(1, maxiter + 1):
        Ap = normal_operator_dask(p, u, v, l, m, chunk_size)

        alpha = rsold / np.vdot(p, Ap).real
        I += alpha * p
        r -= alpha * Ap

        rsnew = np.vdot(r, r).real

        if verbose:
            print(f"CG iter {k:02d} | residual = {np.sqrt(rsnew):.3e}")

        if np.sqrt(rsnew) < tol:
            break

        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return I.real
