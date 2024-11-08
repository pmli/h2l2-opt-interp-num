# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Parametric Penzl example

# %% [markdown]
# ## Imports

# %%
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spint
import scipy.linalg as spla
import scipy.optimize as spopt
import scipy.sparse as sps
from pymor.algorithms.sylvester import solve_sylv_schur
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.defaults import set_defaults
from pymor.models.iosys import LTIModel
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional

# %%
set_defaults(
    {
        'pymor.tools.plot.adaptive.angle_tol': 1,
        'pymor.tools.plot.adaptive.min_rel_dist': 1e-3,
    }
)

# %%
quad_opts = dict(epsabs=0, epsrel=1e-12, limit=1000)

# %% [markdown]
# ## Full-order model

# %%
A1 = np.array([[-1, 0], [0, -1]])
A2 = np.diag(np.arange(-1, -11, -1))
A0 = spla.block_diag(A1, A2)
n = A0.shape[0]
Ap = sps.lil_array((n, n))
Ap[0, 1] = 1
Ap[1, 0] = -1
Ap = Ap.tocsc()
B = np.ones((n, 1))
B[:2, 0] = 5
C = B.T

Aop = NumpyMatrixOperator(A0) + ProjectionParameterFunctional('p') * NumpyMatrixOperator(Ap)
Bop = NumpyMatrixOperator(B)
Cop = NumpyMatrixOperator(C)

fom = LTIModel(Aop, Bop, Cop)

# %%
fom

# %%
p_list = np.geomspace(1, 100, 3)

# %%
fig, ax = plt.subplots()
markers = '.x+'
for p, m in zip(p_list, markers):
    poles = fom.poles(p)
    ax.plot(poles.real, poles.imag, m, label=f'$p$ = {p}')
_ = ax.legend()

# %%
w = (1e-1, 1e4)
fig, ax = plt.subplots()
linestyles = ['-', '--', ':']
for p, ls in zip(p_list, linestyles):
    fom.transfer_function.mag_plot(w, ax=ax, mu=p, linestyle=ls, label=f'$p$ = {p}')
ax.set_title('FOM magnitude plot')
_ = plt.legend()


# %%
def h2l2_norm(m, p_min, p_max):
    def f(p):
        return m.h2_norm(mu=p) ** 2

    res = spint.quad(f, p_min, p_max, **quad_opts)
    return np.sqrt(res[0])


# %%
p_min, p_max = 1, 100

# %%
fom_h2l2_norm = h2l2_norm(fom, p_min, p_max)
fom_h2l2_norm

# %% [markdown]
# ## Initial reduced-order model

# %%
r = 3
A0r0 = A0[:r, :r]
Apr0 = Ap[:r, :r]
Br0 = B[:r, :]
Cr0 = C[:, :r]
# Br0 = np.array([[2], [0], [2], [0], [2], [0], [1], [1]])
# Cr0 = B.T[:, :r] * C[:, :r]
# Cr0[0, 1:6:2] = 0
Ar0op = NumpyMatrixOperator(A0r0) + ProjectionParameterFunctional('p') * NumpyMatrixOperator(Apr0)
Br0op = NumpyMatrixOperator(Br0)
Cr0op = NumpyMatrixOperator(Cr0)
rom0 = LTIModel(Ar0op, Br0op, Cr0op)

# %%
err0 = fom - rom0

# %%
rom0_h2l2_rel_err = h2l2_norm(err0, p_min, p_max) / fom_h2l2_norm
rom0_h2l2_rel_err

# %%
fig, ax = plt.subplots()
for p, ls in zip(p_list, linestyles):
    rom0.transfer_function.mag_plot(w, ax=ax, mu=p, ls=ls, label=f'$p$ = {p}')
ax.set_title('Initial ROM magnitude plot')
_ = plt.legend()

# %%
fig, ax = plt.subplots()
for p, ls in zip(p_list, linestyles):
    err0.transfer_function.mag_plot(w, ax=ax, mu=p, ls=ls, label=f'$p$ = {p}')
ax.set_title('Initial ROM error magnitude plot')
_ = plt.legend()


# %%
def f(s_a, s_b, sigma_a, sigma_b, p_min, p_max):
    d_a = s_a - sigma_a
    d_b = s_b - sigma_b
    if d_a == d_b:
        return (p_max - p_min) / d_a
    return (p_max - p_min) * np.log(d_b / d_a) / (d_b - d_a)


def dfa(s_a, s_b, sigma_a, sigma_b, p_min, p_max):
    d_a = s_a - sigma_a
    d_b = s_b - sigma_b
    if d_a == d_b:
        return -(p_max - p_min) / d_a**2
    return (p_max - p_min) / (d_b - d_a) * (np.log(d_b / d_a) / (d_b - d_a) - 1 / d_a)


def dfb(s_a, s_b, sigma_a, sigma_b, p_min, p_max):
    d_a = s_a - sigma_a
    d_b = s_b - sigma_b
    if d_a == d_b:
        return -(p_max - p_min) / d_a**2
    return (p_max - p_min) / (d_b - d_a) * (-np.log(d_b / d_a) / (d_b - d_a) + 1 / d_b)


def modified_tf(v_a, v_b, res, p_min, p_max):
    def G(s_a, s_b):
        return np.sum([f(s_a, s_b, v1, v2, p_min, p_max) * r for v1, v2, r in zip(v_a, v_b, res)])

    def dGa(s_a, s_b):
        return np.sum([dfa(s_a, s_b, v1, v2, p_min, p_max) * r for v1, v2, r in zip(v_a, v_b, res)])

    def dGb(s_a, s_b):
        return np.sum([dfb(s_a, s_b, v1, v2, p_min, p_max) * r for v1, v2, r in zip(v_a, v_b, res)])

    return G, dGa, dGb


# %%
fom_poles_min = np.concatenate(
    (
        np.array([-1 + p_min * 1j, -1 - p_min * 1j]),
        -np.arange(1, 11),
    )
)
fom_poles_max = np.concatenate(
    (
        np.array([-1 + p_max * 1j, -1 - p_max * 1j]),
        -np.arange(1, 11),
    )
)
fom_res = B[:, 0] * C[0, :]
fom_G, fom_dGa, fom_dGb = modified_tf(fom_poles_min, fom_poles_max, fom_res, p_min, p_max)

# %%
rom0_poles_min = np.concatenate(
    (
        np.array([-1 + p_min * 1j, -1 - p_min * 1j]),
        [-1],
    )
)
rom0_poles_max = np.concatenate(
    (
        np.array([-1 + p_max * 1j, -1 - p_max * 1j]),
        [-1],
    )
)
rom0_res = Br0[:, 0] * Cr0[0, :]
rom0_G, rom0_dGa, rom0_dGb = modified_tf(rom0_poles_min, rom0_poles_max, rom0_res, p_min, p_max)

# %%
[
    abs(fom_G(-v1.conj(), -v2.conj()) - rom0_G(-v1.conj(), -v2.conj()))
    / abs(fom_G(-v1.conj(), -v2.conj()))
    for v1, v2 in zip(rom0_poles_min, rom0_poles_max)
]

# %%
[
    abs(fom_dGa(-v1.conj(), -v2.conj()) - rom0_dGa(-v1.conj(), -v2.conj()))
    / abs(fom_dGa(-v1.conj(), -v2.conj()))
    for v1, v2 in zip(rom0_poles_min, rom0_poles_max)
]

# %%
[
    abs(fom_dGb(-v1.conj(), -v2.conj()) - rom0_dGb(-v1.conj(), -v2.conj()))
    / abs(fom_dGb(-v1.conj(), -v2.conj()))
    for v1, v2 in zip(rom0_poles_min, rom0_poles_max)
]

# %% [markdown]
# ## Diagonal ROM optimization

# %% [markdown]
# ROM:
#
# $$
# \hat{A} =
# \begin{bmatrix}
#   a_1 + p b_1 & c_1 + p d_1 \\
#   -c_1 - p d_1 & a_1 + p b_1 \\
#   & & a_2 + p b_2
# \end{bmatrix}, \quad
# \hat{B} =
# \begin{bmatrix}
#   2 \\
#   0 \\
#   1
# \end{bmatrix}, \quad
# \hat{C} =
# \begin{bmatrix}
#   y_1 & y_2 & y_3
# \end{bmatrix}
# $$


# %%
def vec_to_rom(x):
    a, b, c, d, y = np.split(x, np.cumsum([2, 2, 1, 1]))

    A0 = np.array([[a[0], c[0], 0], [-c[0], a[0], 0], [0, 0, a[1]]])
    Ap = np.array([[b[0], d[0], 0], [-d[0], b[0], 0], [0, 0, b[1]]])
    B = np.array([[2], [0], [1]])
    C = y[np.newaxis, :]

    Aop = NumpyMatrixOperator(A0) + ProjectionParameterFunctional('p') * NumpyMatrixOperator(Ap)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)

    rom = LTIModel(Aop, Bop, Cop)

    return rom


def obj(x):
    rom = vec_to_rom(x)

    if rom.poles(p_min).real.max() >= 0 or rom.poles(p_max).real.max() >= 0:
        return 0

    def f(p):
        mu = rom.parameters.parse(p)
        Ph = rom.gramian('c_dense', mu=mu)
        Pt = solve_sylv_schur(fom.A.assemble(mu=mu), rom.A.assemble(mu=mu), B=fom.B, Br=rom.B)
        Cr = to_matrix(rom.C)
        return np.trace(Cr @ Ph @ Cr.T - 2 * fom.C.apply(Pt.lincomb(Cr)).to_numpy().T)

    res = spint.quad(f, p_min, p_max, **quad_opts)
    return res[0]


def jac(x):
    rom = vec_to_rom(x)

    if rom.poles(p_min).real.max() >= 0 or rom.poles(p_max).real.max() >= 0:
        return np.ones_like(x)

    def f(p):
        mu = rom.parameters.parse(p)
        Ph = rom.gramian('c_dense', mu=mu)
        Qh = rom.gramian('o_dense', mu=mu)
        Pt, Qt = solve_sylv_schur(
            fom.A.assemble(mu=mu),
            rom.A.assemble(mu=mu),
            B=fom.B,
            Br=rom.B,
            C=fom.C,
            Cr=rom.C,
        )
        Cr = to_matrix(rom.C)
        grad_Ar0 = 2 * (Qh @ Ph - Qt.inner(Pt))
        grad_Arp = p * grad_Ar0
        grad_Cr = 2 * (Cr @ Ph - fom.C.apply(Pt).to_numpy().T)
        d = grad_Ar0.diagonal()
        grad_a = [(d[0] + d[1]) / 2, d[2]]
        d = grad_Arp.diagonal()
        grad_b = [(d[0] + d[1]) / 2, d[2]]
        grad_c = [(grad_Ar0[0, 1] - grad_Ar0[1, 0]) / 2]
        grad_d = [(grad_Arp[0, 1] - grad_Arp[1, 0]) / 2]
        grad_y = grad_Cr[0, :]
        grad = np.concatenate((grad_a, grad_b, grad_c, grad_d, grad_y))
        return grad

    res = spint.quad_vec(f, p_min, p_max, **quad_opts)
    return res[0]


# %%
cr0 = Br0[:, 0] * Cr0[0, :]
cr0[1] = 0
x0 = np.concatenate(
    (
        rom0.A.operators[0].matrix.diagonal()[::2],
        rom0.A.operators[1].matrix.diagonal()[::2],
        [rom0.A.operators[0].matrix[0, 1]],
        [rom0.A.operators[1].matrix[0, 1]],
        cr0,
    )
)
tic = perf_counter()
res = spopt.minimize(
    obj,
    x0,
    jac=jac,
    method='BFGS',
    options={'disp': True, 'gtol': 1e-5},
)
elapsed = perf_counter() - tic
print(f'Elapsed time: {elapsed}')

# %%
res

# %%
rom = vec_to_rom(res.x)

# %%
rom

# %%
print(rom.A.operators[0].matrix)

# %%
print(rom.A.operators[1].matrix)

# %%
rom.B.matrix

# %%
rom.C.matrix

# %%
err = fom - rom

# %%
rom_h2l2_rel_err = h2l2_norm(err, p_min, p_max) / fom_h2l2_norm
rom_h2l2_rel_err


# %%
rom0_poles_min

# %%
rom0_poles_max

# %%
rom0_res

# %%
ar = rom.A.assemble(mu=fom.parameters.parse(p_min)).matrix.diagonal()[:1]
br = rom.A.assemble(mu=fom.parameters.parse(p_min)).matrix.diagonal(1)[:1]
cr = rom.A.assemble(mu=fom.parameters.parse(p_min)).matrix.diagonal()[-1:]
rom_poles_min = np.concatenate((ar + 1j * br, ar - 1j * br, cr))

ar = rom.A.assemble(mu=fom.parameters.parse(p_max)).matrix.diagonal()[:1]
br = rom.A.assemble(mu=fom.parameters.parse(p_max)).matrix.diagonal(1)[:1]
cr = rom.A.assemble(mu=fom.parameters.parse(p_max)).matrix.diagonal()[-1:]
rom_poles_max = np.concatenate((ar + 1j * br, ar - 1j * br, cr))

ar, br, cr = rom.C.matrix[0, :]
rom_res = np.array([ar + 1j * br, ar - 1j * br, cr])

rom_G, rom_dGa, rom_dGb = modified_tf(rom_poles_min, rom_poles_max, rom_res, p_min, p_max)

# %%
rom_poles_min

# %%
rom_poles_max

# %%
rom_res

# %%
[
    abs(fom_G(-v1.conj(), -v2.conj()) - rom_G(-v1.conj(), -v2.conj()))
    / abs(fom_G(-v1.conj(), -v2.conj()))
    for v1, v2 in zip(rom_poles_min, rom_poles_max)
]

# %%
[
    abs(fom_dGa(-v1.conj(), -v2.conj()) - rom_dGa(-v1.conj(), -v2.conj()))
    / abs(fom_dGa(-v1.conj(), -v2.conj()))
    for v1, v2 in zip(rom_poles_min, rom_poles_max)
]

# %%
[
    abs(fom_dGb(-v1.conj(), -v2.conj()) - rom_dGb(-v1.conj(), -v2.conj()))
    / abs(fom_dGb(-v1.conj(), -v2.conj()))
    for v1, v2 in zip(rom_poles_min, rom_poles_max)
]

# %%
fig, ax = plt.subplots()
for p, m in zip(p_list, markers):
    poles = rom.poles(p)
    ax.plot(poles.real, poles.imag, m, label=f'$p$ = {p}')
_ = ax.legend()

# %%
fig, ax = plt.subplots()
for p, ls in zip(p_list, linestyles):
    rom.transfer_function.mag_plot(w, ax=ax, mu=p, ls=ls, label=f'$p$ = {p}')
ax.set_title('ROM magnitude plot')
_ = plt.legend()

# %%
fig, ax = plt.subplots()
for p, ls in zip(p_list, linestyles):
    err.transfer_function.mag_plot(w, ax=ax, mu=p, ls=ls, label=f'$p$ = {p}')
ax.set_title('ROM error magnitude plot')
_ = plt.legend()
