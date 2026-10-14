# Must run before any MuJoCo Warp kernel is compiled (see src/compat).
import src.compat  # noqa: F401
from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]

import_packages(__name__, _BLACKLIST_PKGS)
