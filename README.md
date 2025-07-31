# Robust and replicable effects of ageing on resting state brain electrophysiology measured with MEG

## Installation

Dependencies are contained within `pyproject.toml` and recommend using `uv` for installation. The following code creates a virtual environment with exact specifications in current directory (likely code root)

```
uv venv
source .venv/bin/activate
uv pip sync pyproject.toml
```

Run above once to install, then activate in future sessions with

```
source .venv/bin/activate
```

UoB specific:

```
module load bear-apps/2023a
module load uv/0.6.5
module load ImageMagick/7.1.1-15-GCCcore-12.3.0
```


## Set-up

Update paths in `pyproject_paths.toml` to point to data directories and output locations.


## Functions

Preprocessing and (optional) SLURM cluster submission scripts.
 
```
bigmeg-0-preproc_oxford_preproc-slurm.sh
bigmeg-0-preproc_oxford_preproc-slurm.py
bigmeg-0-preproc_nottingham_preproc-slurm.sh
bigmeg-0-preproc_nottingham_preproc-slurm.py
bigmeg-0-preproc_camcan_preproc-stats.py
bigmeg-0-preproc_camcan_preproc-slurm.sh
bigmeg-0-preproc_camcan_preproc-slurm.py
bigmeg-0-preproc_cambridge_preproc-slurm.sh
bigmeg-0-preproc_cambridge_preproc-slurm.py
```

Data analysis

```
bigmeg-1a_group-glm-perms_age-only.py
bigmeg-1b_group-glm-perms_age-with-covariates.py
bigmeg-1c_group-glm-perms_age-only-variant-preproc.py
bigmeg-1d_group-glm-perms_age-only-variant-headpos.py
```

Figures

```
bigmeg-3a_figures123_camcan-age-only.py
bigmeg-3b_figures45_alldata-age-only.py
bigmeg-3c_figures6_camcan-age-only-absolute.py
bigmeg-3d_figures78_camcan-age-covariates.py
```