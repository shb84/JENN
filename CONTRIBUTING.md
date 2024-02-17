# Contribution 

A distinction is made between the `dev` environment and the `ci` environment. The former is generated from the 
`environment.yml` file and serves as the local environment in which to try things out. The latter is generated 
from `conda-lock` files that define the frozen specs of the build environment (located in `deploy/specs`). 
Hence, after local decisions are made in the `dev` environment, the lock files must be updated accordingly for testing under `ci`. 

--- 
## Installation

#### `dev`
Assuming [conda](https://conda.org/) is installed:

```bash
conda env update --file environment.yml --name jenn
conda activate jenn
pip install -e .
pytest
```

All tests should pass. 

#### `ci` 

Commands are run with [`doit`](https://pydoit.org/) defined in a `dodo.py` file. To see all `doit` commands:

```bash
doit list --all --status
```

---
## Procedure

The recommended process is to do all local work in the `dev` environment. Upon satisfaction, before merge requests, follow the steps below which ensure `ci` will pass. 

#### Step 1: Update Environment Specs (optional)

_**IF AND ONLY IF** the `environment.yml` file was updated during development, then the `ci` environment must also be updated accordingly. To do so, update `deploy/specs/*.yml` accordingly and re-generate the lock files_: 
 
```bash
doit lock
```

#### Step 2: Run Unit Tests

_Make sure the unit tests are passing_: 

```bash
doit install test
```

#### Step 3: Fix Lint Issues 

_Make sure the code is well formatted. Fix manually if needed_: 

```bash
doit fix lint
```

#### Step 4: Run Notebooks (optional) 

_If applicable, mannually check notebooks in `ci` environment_: 

```bash
doit lab
```

#### Step 5: Pull Request

_Create a draft PR on GitHub in order to trigger CI and push changes_: 

```bash
git add -u 
git status 
git commit -m "description"
git push
```

When satisfied, change PR status to ready for review. 

--- 
## License
Distributed under the terms of the MIT License.
