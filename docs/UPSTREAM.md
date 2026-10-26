# Source and branch history

This branch uses the full mjlab framework as its main project. It is installed
from `src/mjlab` in this checkout; it does not install the old Unitree wrapper or
require a second mjlab source checkout.

| Item | Revision or location |
| --- | --- |
| This branch | `codex/mjlab-upstream` |
| Requested starting commit | `1c4e48c1721d2388793ebd24e783c51858214ead` |
| Imported source | [mujocolab/mjlab](https://github.com/mujocolab/mjlab) `main` |
| Imported commit | [`8bb2bc9e187797689ec82c78c35019afd2fd0376`](https://github.com/mujocolab/mjlab/commit/8bb2bc9e187797689ec82c78c35019afd2fd0376) |
| Local additions | Root `env.sh`, `scripts/pace/`, `docs/pace/`, this page, the teaching README, setup tests, and changelog entries |

The import commit has the requested starting commit as its first parent and the
upstream mjlab commit as its second parent. The original Unitree project files
are replaced by the upstream tree. The upstream license, assets, tests,
documentation, package configuration, and dependency lock are retained.

The source revision is from upstream **main**, not the `v1.6.0` release tag.
Although its package version is still `1.6.0`, its lock uses newer dependencies,
including MuJoCo 3.16 and RSL-RL 5.5.1. Use this branch's `uv.lock` and its `cu130`
extra on PACE. The old `requirements-pace.lock.txt`, `setup.py`, `src.tasks`, and
`python scripts/train.py Unitree-G1-Flat` instructions do not apply.

## Unitree support

Upstream already includes G1 and Go1 velocity tasks and G1 motion tracking.
List the available tasks with `uv run list-envs --keyword Unitree`.
The PACE examples use `Mjlab-Velocity-Flat-Unitree-G1`.

The earlier Unitree implementation, extra robot variants, and hardware deployment
code remain accessible at the requested
[base commit](https://github.com/wognl0402/unitree_rl_mjlab/tree/1c4e48c1721d2388793ebd24e783c51858214ead).
They are not copied into the new package: port any additional robot or task
against the current API and test it before use. The existing remote branches
are left intact.

## Update from upstream

In a clean checkout of this branch, add a separate remote for mjlab:

```bash
git remote add mjlab https://github.com/mujocolab/mjlab.git
git fetch mjlab main
git merge mjlab/main
```

If `mjlab` already exists, inspect `git remote -v` and reuse it. The old
`upstream` remote may point at Unitree; do not use it for framework updates.
The preserved merge history lets Git identify changes after the imported mjlab
revision. Review lockfile and API changes, update this page's imported revision,
run the development checks, and repeat the README's PACE GPU smoke job before
publishing.

No default branch or existing branch has been rewritten.
