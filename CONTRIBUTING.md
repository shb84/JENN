# CONTRIBUTING 

Contibutions are welcome. Thank you for helping make the project better! 

## Installation

This project uses [`pixi`](https://pixi.sh/latest/), which must be installed. 

__Linux & macOS__ 
```
curl -fsSL https://pixi.sh/install.sh | bash
``` 

__Windows__
```
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

_That's it. You are now ready to go!_

## Running 

To display a list of available tasks, type `pixi run`. 

__Example usage__
```
pixi run test
```
OR
```
pixi run lab
```

## Making Changes

- [ ] Fork the repo 
- [ ] Make code changes 
- [ ] Update package dependencies in `pyproject.toml` as necessary 
- [ ] Update project dependencies in `pixi.toml` as necessary
- [ ] Ensure QA passes locally: `pixi run all`
- [ ] Commit and push changes to GitHub (this automatically triggers CI)
- [ ] Create pull request when ready

## Release 

*Only project owners and administrators can make new releases.* 

A new release is created by pushing a new tag to the remote (e.g. `v1.0.9`). This triggers a __test-build-deploy__ workflow that publishes to `pypi.org`, `GitHub Pages` and `Github Release`. Tag pattern must be `v*`.

### Prerequisites

Either [pypi](https://pypi.org/) and [testpypi](https://test.pypi.org/) need to be setup for [trusted publishing](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/) or Github must be provided an [API token](https://pypi.org/help/#apitoken) to enable communication between these servers. Currently, the latter is used.

### Mock Release

It's a good idea to do a mock release using `testpypi` from local install: 

```bash
pixi run testpypi
```

_Check that the package appears on `testpypi` and try manually installing it in a fresh virtual environment to ensure it runs as expected, as an extra layer of precaution. If not, please help update the CI procedure to catch the newly found issues._

### Procedure 
Assuming `master` is locally up-to-date, update version number in `src/jenn/__init__.py`: 

```bash
__version__ = "1.0.9"
```

Push the version change to the remote: 

```bash
git add -u 
git commit -m "changed version to v1.0.9"
git push 
```

Tag commit for release and push to trigger release pipeline: 

```bash
git tag v1.0.9
git push origin v1.0.9
```

Once the pipeline has succeeded, check that there is now a new release on `pypi.org`, `GitHub Pages` and `Github Release`. 

### TestPyPi

- [ ] In account settings on [testpypi.org](https://test.pypi.org/), go to the API tokens section and select "Add API token" 
- [ ] Then use that token to configure your local [`.pyprc`](https://packaging.python.org/en/latest/specifications/pypirc/) file: 

```bash
[distutils]
index-servers =
    testpypi

[testpypi]
repository: https://test.pypi.org/legacy/
username: __token__
password: pypi-...
```

