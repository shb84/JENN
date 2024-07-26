# CONTRIBUTING 

Contibutions are welcome. Thank you for helping make the project better! 

--- 
## Installation

This project relies on [`pixi`](https://pixi.sh/latest/), which needs to be installed: 

```
curl -fsSL https://pixi.sh/install.sh | bash
``` 

That's it! You are now ready to go. 

---
## Running 

Just use `pixi` to run tasks (defined in `pyproject.toml`) command, e.g.: 
```
pixi run test
```
OR
```
pixi run lab
```

---
## Making Changes

### Step 1: Update dependencies 

If needed, update any project dependencies in the `pyproject.toml` (they will automatically be picked up by `pixi`):

```
[project]
dependencies = [
  "jsonpointer>=2.4",
  "jsonschema>=4.22",
  "orjson>=3.9",
  "numpy>=1.22",
]

[project.optional-dependencies]
plot = [
  "matplotlib",
]
```

### Step 2: Run Unit Tests

Make sure the unit tests are passing: 

```bash
pixi run test
```

### Step 3: Fix Lint Issues 

Make sure the code is well formatted (fix manually if needed): 

```bash
pixi run lint
```

### Step 4: Test Build

Test docs are building locally: 

```bash
pixi run build-docs
```

Test distribution is building locally: 

```bash
pixi run build-dist
```

### Step 5: Test Release

Test release on `testpypi`: 

```bash
pixi run release
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

### Step 5: Commit / Pull Request

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

The release process has been automated via Github Actions. In summary, a new release is created by pushing a new tag to the remote (e.g. `v1.0.8`), which triggers a _test-build-deploy_ workflow that publishes to `pypi.org`, `GitHub Pages` and `Github Release`. Tag pattern must be `v*`.

> NOTE: 
> either [pypi](https://pypi.org/) and [testpypi](https://test.pypi.org/) need to be setup for [trusted publishing](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/) or Github must be provided an [API token](https://pypi.org/help/#apitoken) to enable communication between GitHub, TestPyPI, and PyPI. Currently, the latter is used.

Assuming `master` is locally up-to-date, manually update the pyproject.toml version number:

```bash
version = "1.0.8"
```

For now, this must also be done manually in `src/jenn/__init__.py`: 

```bash
__version__ = "1.0.8"
```

Push the version change to the remote: 

```bash
git add -u 
git commit -m "changed version to v1.0.8"
git push 
```

Tag commit for release and push to trigger release pipeline: 

```bash
git tag v1.0.8
git push origin v1.0.8
```

Once the pipeline has succeeded, check that there is now a new release on `pypi.org`, `GitHub Pages` and `Github Release`. If so, the last step is to lock the branch associated with the release, so it can be easily accessed but not changed. 

```bash
git checkout -b jenn-v1.0.8 
git push --set-upstream origin jenn-v1.0.8 
```

On GitHub, go to Project `Settings > Branches` and add a "Lock branch" rule for `jenn-v1.0.8`.

