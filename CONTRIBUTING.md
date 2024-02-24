# CONTRIBUTING 

Contibutions are welcome. Thank you for helping make the project better! 

--- 
## Installation

### `dev` environment

_The `dev` environment serves as the local environment in which to try things._ 

Assuming [conda](https://conda.org/) is installed:

```bash
conda env update --file environment.yml --name jenn
conda activate jenn
pip install -e .
pytest
```

All tests should pass. 

### `ci` environment

_The `ci` environment serves as the remote environment used by Github Actions. It is generated from frozen specs with `conda-lock`. Commands are run with [`doit`](https://pydoit.org/) (defined in the `dodo.py` file)._ 

To see all `doit` commands:

```bash
doit list --all --status
```

---
## Making Changes

Do all local work in the `dev` environment. Upon satisfaction, before making a commit or merge request, manually follow the steps below to improve chances of passing `ci` (which are the same steps run in `ci`). 

### Step 1: Update Environment Specs (optional)

_**IF** the `environment.yml` file was updated, then the `ci` environment must also be updated. To do so, modify `deploy/specs/*.yml` accordingly and re-generate the lock files_: 
 
```bash
doit lock
```

### Step 2: Run Unit Tests

Make sure the unit tests are passing in the `ci` environment: 

```bash
doit test
```

### Step 3: Fix Lint Issues 

Make sure the code is well formatted (fix manually if needed): 

```bash
doit fix lint
```

### Step 4: Run Notebooks (optional) 

If applicable, mannually check notebooks in `ci` environment: 

```bash
doit lab
```

### Step 5: Test Release (optional)

Test release on `testpypi`: 

```bash
doit build release
```

This will require creating an API token on `testpypi`: 

* In your account settings, go to the API tokens section and select "Add API token" 

Then use that token to configure your local [`.pyprc`](https://packaging.python.org/en/latest/specifications/pypirc/) file: 

```bash
[distutils]
index-servers =
    testpypi

[testpypi]
repository: https://test.pypi.org/legacy/
username: __token__
password: pypi-...
```

### Step 6: Commit / Pull Request

Commit changes. CI is triggered on push: 

```bash
git add -u 
git status 
git commit -m "description"
git push
```

_Create a PR on GitHub when ready_ 

## Release 

**Only project owners and administrators can make new releases.** 

The release process has been automated via Github Actions. In summary, a new release is created by pushing a new tag to the remote (e.g. `v1.0.0`), which triggers a _test-build-deploy_ workflow that publishes to `pypi.org`, `GitHub Pages` and `Github Release`. Tag pattern must be `v*`.

> NOTE: 
> either [pypi](https://pypi.org/) and [testpypi](https://test.pypi.org/) need to be setup for [trusted publishing](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/) or Github must be provided an [API token](https://pypi.org/help/#apitoken) to enable communication between GitHub, TestPyPI, and PyPI. Currently, the latter is used.

### Steps 

Assuming `master` is locally up-to-date, manually update the pyproject.toml version number:

```bash
version = "1.0.0"
```

Push the version change to the remote: 

```bash
git add -u 
git commit -m "changed version to v1.0.0"
git push 
```

Tag commit for release and push to trigger release pipeline: 

```bash
git tag v1.0.0
git push origin v1.0.0
```

Once the pipeline has succeeded, check that there is now a new release on `pypi.org`, `GitHub Pages` and `Github Release`. If so, the last step is to lock the branch associated with the release, so it can be easily accessed but not changed. 

```bash
git checkout -b jenn-v1.0.0 
git push --set-upstream origin jenn-v1.0.0 
```

On GitHub, go to Project `Settings > Branches` and add a "Lock branch" rule for `jenn-v1.0.0`.

