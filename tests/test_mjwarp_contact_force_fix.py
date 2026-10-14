"""Regression tests for the MuJoCo Warp 3.5.0 pyramidal contact-force decoding fix (src/compat).

The decoder bug only shows when a contact's pyramid rows are non-contiguous in efc_force,
which on GPU depends on thread scheduling. These tests construct that layout explicitly:
the true rows of a CPU MuJoCo contact are scattered over a constraint-force array filled
with unrelated (large) forces, then decoded with the upstream and the fixed function.

Run: python -m pytest tests/test_mjwarp_contact_force_fix.py -q
"""

import mujoco
import numpy as np
import pytest
import warp as wp

import src.compat  # noqa: F401  (applies the fix)
from mujoco_warp._src import support
from mujoco_warp._src.types import ConeType, vec5
from src.compat import mujoco_warp_contact_force as fix

FIXED = fix.contact_force_fn
ORIGINAL = fix.original_contact_force_fn()
NJMAX = 16
FILLER = 50.0  # N, force of unrelated constraint rows (e.g. a foot contact)

_XML = """
<mujoco>
  <option cone="pyramidal" timestep="0.005"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1" friction="1 0.005 0.0001"/>
    <body pos="0.0 0 0.042">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="2" condim="3" friction="1 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""


def _cpu_contact():
  """A real pyramidal condim-3 contact: (frame, friction, 4 efc rows, mj_contactForce)."""
  m = mujoco.MjModel.from_xml_string(_XML)
  d = mujoco.MjData(m)
  d.qvel[0] = 0.3  # sliding, so the tangential force is nonzero
  mujoco.mj_forward(m, d)
  c = d.contact[0]
  assert c.dim == 3 and m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
  rows = d.efc_force[c.efc_address:c.efc_address + 4].copy()
  f6 = np.zeros(6)
  mujoco.mj_contactForce(m, d, 0, f6)
  assert f6[0] > 1.0
  return c.frame.reshape(3, 3).copy(), np.array(c.friction, dtype=np.float32), rows, f6


@wp.kernel
def _decode_fixed(frame: wp.array(dtype=wp.mat33), friction: wp.array(dtype=vec5), dim: wp.array(dtype=int),
                  efc_address: wp.array2d(dtype=int), efc_force: wp.array2d(dtype=float), njmax: int,
                  nacon: wp.array(dtype=int), out: wp.array(dtype=wp.spatial_vector)):
  out[0] = FIXED(int(ConeType.PYRAMIDAL), frame, friction, dim, efc_address, efc_force, njmax, nacon, 0, 0, False)


@wp.kernel
def _decode_original(frame: wp.array(dtype=wp.mat33), friction: wp.array(dtype=vec5), dim: wp.array(dtype=int),
                     efc_address: wp.array2d(dtype=int), efc_force: wp.array2d(dtype=float), njmax: int,
                     nacon: wp.array(dtype=int), out: wp.array(dtype=wp.spatial_vector)):
  out[0] = ORIGINAL(int(ConeType.PYRAMIDAL), frame, friction, dim, efc_address, efc_force, njmax, nacon, 0, 0, False)


def _decode(kernel, frame, friction, rows, addresses, dim=3):
  efc = np.full((1, NJMAX), FILLER, dtype=np.float32)
  for k, a in enumerate(addresses):
    if a >= 0:
      efc[0, a] = rows[k]
  args = [
    wp.array([frame], dtype=wp.mat33, device="cpu"), wp.array([friction], dtype=vec5, device="cpu"),
    wp.array([dim], dtype=int, device="cpu"), wp.array(np.array([addresses], dtype=np.int32), dtype=int, device="cpu"),
    wp.array(efc, dtype=float, device="cpu"), NJMAX, wp.array([1], dtype=int, device="cpu"),
  ]
  out = wp.zeros(1, dtype=wp.spatial_vector, device="cpu")
  wp.launch(kernel, dim=1, inputs=args + [out], device="cpu")
  return out.numpy()[0]


# Tolerance: decoding uses float32 storage of the CPU rows; measured Warp-vs-CPU agreement on
# the full simulation was 0.06 N median, 0.43 N max (solver differences, absent here).
TOL = 1e-3


def test_fix_is_installed():
  assert fix.is_active()
  assert support.contact_force_fn is FIXED


def test_contiguous_rows_decoded_correctly_by_both():
  frame, mu, rows, f6 = _cpu_contact()
  for kernel in (_decode_fixed, _decode_original):
    out = _decode(kernel, frame, mu, rows, [3, 4, 5, 6])
    np.testing.assert_allclose(out[:3], f6[:3], atol=TOL)


def test_noncontiguous_rows_fixed_matches_mj_contactForce():
  frame, mu, rows, f6 = _cpu_contact()
  addresses = [9, 2, 13, 6]  # interleaved with unrelated constraint rows
  fixed = _decode(_decode_fixed, frame, mu, rows, addresses)
  np.testing.assert_allclose(fixed[:3], f6[:3], atol=TOL)
  assert np.all(np.isfinite(fixed))


def test_noncontiguous_rows_original_decoder_is_wrong():
  """Documents the upstream bug: this layout is decoded from the wrong rows."""
  frame, mu, rows, f6 = _cpu_contact()
  original = _decode(_decode_original, frame, mu, rows, [9, 2, 13, 6])
  assert abs(original[0] - f6[0]) > 10.0  # reads FILLER rows starting at address 9


def test_no_contact_rows_give_zero_force():
  frame, mu, rows, _ = _cpu_contact()
  out = _decode(_decode_fixed, frame, mu, rows, [-1, -1, -1, -1])
  assert np.all(out == 0.0)


def test_condim1_uses_single_row():
  frame, mu, rows, _ = _cpu_contact()
  out = _decode(_decode_fixed, frame, mu, rows, [5, -1, -1, -1], dim=1)
  assert out[0] == pytest.approx(rows[0], abs=TOL) and np.all(out[1:] == 0.0)
