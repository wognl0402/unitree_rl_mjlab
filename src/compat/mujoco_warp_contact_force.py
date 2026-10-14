"""Compatibility fix: MuJoCo Warp 3.5.0 pyramidal contact-force decoding.

Upstream bug (mujoco_warp/_src/support.py, ``contact_force_fn`` -> ``_decode_pyramid``):
for a pyramidal-cone contact the decoder reads the friction-pyramid rows at
``efc_address[c, 0] + k``, i.e. it assumes the contact's 2*(condim-1) constraint
rows are contiguous. They are not: ``constraint.py`` allocates every row with its
own ``wp.atomic_add`` on the world's ``nefc`` counter and stores each row address
separately in ``contact.efc_address[c, k]``. When rows interleave with other
constraints the decoder sums unrelated rows (e.g. a foot contact's forces).

Physics is unaffected (constraint forces reach the dynamics through
``qfrc_constraint``). Everything that *decodes* contact forces is affected:
contact sensors (netforce / maxforce / none), ``cfrc_ext`` (rne_postconstraint)
and the ``contact_force`` kernel. All of them call ``support.contact_force_fn``,
which Warp resolves when it generates kernel code, so rebinding that one
function before any kernel is compiled fixes every consumer.

The corrected function reads row k from ``efc_address[c, k]``. It is otherwise
identical to upstream (same signature, frames, signs, elliptic branch).

Remove this module once MuJoCo Warp ships a fix. Set
UNITREE_MJWARP_PYRAMID_FIX=0 to disable it (reproduces pre-fix behavior).
"""

from __future__ import annotations

import importlib.metadata
import inspect
import os
import warnings

import warp as wp
from mujoco_warp._src import support
from mujoco_warp._src.types import ConeType, vec5

AFFECTED_VERSIONS = ("3.5.0",)
_BUGGY_SIGNATURE = "adr = 2 * i + efc_address"
_ORIGINAL = support.contact_force_fn
_applied = False


@wp.func
def contact_force_fn(
  # Model:
  opt_cone: int,
  # Data in:
  contact_frame_in: wp.array(dtype=wp.mat33),
  contact_friction_in: wp.array(dtype=vec5),
  contact_dim_in: wp.array(dtype=int),
  contact_efc_address_in: wp.array2d(dtype=int),
  efc_force_in: wp.array2d(dtype=float),
  njmax_in: int,
  nacon_in: wp.array(dtype=int),
  # In:
  worldid: int,
  contact_id: int,
  to_world_frame: bool,
) -> wp.spatial_vector:
  """Extract 6D force:torque for one contact, in contact frame by default."""
  force = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  condim = contact_dim_in[contact_id]
  efc_address = contact_efc_address_in[contact_id, 0]

  if contact_id >= 0 and contact_id <= nacon_in[0] and efc_address >= 0:
    if opt_cone == ConeType.PYRAMIDAL:
      if condim == 1:
        force[0] = efc_force_in[worldid, efc_address]
      else:
        mu = contact_friction_in[contact_id]
        for i in range(condim - 1):
          # Each pyramid row has its own address (rows need not be contiguous).
          adr1 = contact_efc_address_in[contact_id, 2 * i]
          adr2 = contact_efc_address_in[contact_id, 2 * i + 1]
          dir1 = float(0.0)
          dir2 = float(0.0)
          if adr1 >= 0 and adr1 < njmax_in:
            dir1 = efc_force_in[worldid, adr1]
          if adr2 >= 0 and adr2 < njmax_in:
            dir2 = efc_force_in[worldid, adr2]
          force[0] += dir1 + dir2
          force[i + 1] = (dir1 - dir2) * mu[i]
    else:
      for i in range(condim):
        if contact_efc_address_in[contact_id, i] < njmax_in:
          force[i] = efc_force_in[worldid, contact_efc_address_in[contact_id, i]]

  if to_world_frame:
    # Transform both top and bottom parts of spatial vector by the full contact frame matrix
    t = wp.spatial_top(force) @ contact_frame_in[contact_id]
    b = wp.spatial_bottom(force) @ contact_frame_in[contact_id]
    force = wp.spatial_vector(t, b)

  return force


def apply() -> bool:
  """Install the corrected decoder. Returns True if the fix is active."""
  global _applied
  if _applied:
    return True
  if os.environ.get("UNITREE_MJWARP_PYRAMID_FIX", "1") == "0":
    warnings.warn("MuJoCo Warp pyramid contact-force fix DISABLED (UNITREE_MJWARP_PYRAMID_FIX=0).")
    return False
  version = importlib.metadata.version("mujoco-warp")
  source = inspect.getsource(support._decode_pyramid.func) + inspect.getsource(support.contact_force_fn.func)
  if version not in AFFECTED_VERSIONS or _BUGGY_SIGNATURE not in source:
    warnings.warn(
      f"mujoco-warp {version}: pyramid contact-force fix not applied (version not in "
      f"{AFFECTED_VERSIONS} or upstream decoder changed). Re-validate and remove src/compat if fixed."
    )
    return False
  support.contact_force_fn = contact_force_fn
  _applied = True
  return True


def is_active() -> bool:
  return _applied and support.contact_force_fn is contact_force_fn


def original_contact_force_fn():
  """The unpatched upstream function (for regression tests)."""
  return _ORIGINAL
