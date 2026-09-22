# Tracking the original Unitree repository

This is a fork of <https://github.com/unitreerobotics/unitree_rl_mjlab>.

| Name | Purpose |
| --- | --- |
| `origin` | Your fork: `wognl0402/unitree_rl_mjlab`; push changes here. |
| `upstream` | Original Unitree repository; fetch its updates here. |
| `upstream-main` | Clean mirror of original Unitree `main`, without PACE changes. |
| `codex/pace-quickstart` | PACE dependency pins, installer, checks, and Viser guide. |
| `main` | Original fork branch, unchanged by the PACE setup commit. |

GitHub knows this repository is a fork. Git remotes and local branch-tracking
settings are local configuration, so each new clone should configure them:

```bash
git clone --branch codex/pace-quickstart \
  https://github.com/wognl0402/unitree_rl_mjlab.git
cd unitree_rl_mjlab
git remote add upstream https://github.com/unitreerobotics/unitree_rl_mjlab.git
git fetch upstream
git branch --track upstream-main upstream/main
git config remote.pushDefault origin
```

If the upstream remote or local branch already exists, inspect `git remote -v`
and `git branch -vv` rather than repeating the creation command. The dedicated
checkout used to publish this branch already has these settings.

## Bring in a new upstream version

Start with a clean working tree:

```bash
git fetch upstream
git switch upstream-main
git merge --ff-only upstream/main
git push origin upstream-main

git switch codex/pace-quickstart
git merge upstream/main
```

Resolve any merge conflicts, review dependency changes, and run the short
training and Viser checks in [the quickstart](PACE_QUICKSTART.md). Then publish:

```bash
git push origin codex/pace-quickstart
```

The mirror updates when these commands are run; no scheduled sync is installed.
Keep `upstream-main` free of local commits. Merging preserves the original
history and the PACE setup commits; no force push is needed.
