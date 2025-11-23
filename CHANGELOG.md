<!--
feat: A new feature.

fix: A bug fix.

docs: Documentation changes.

style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc).

refactor: A code change that neither fixes a bug nor adds a feature.

perf: A code change that improves performance.

test: Changes to the test framework.

build: Changes to the build process or tools.
-->

# Changelog

## v1.0.9 (YYYY-MM-DD)

### Build

- Added `pixi.toml` (rather than putting everything in `pyproject.toml`)
- Updated CI worflow 

### Docs

- Simplified CONTRIBUTING 

## v1.0.8 (2024-06-26)

### Build

- Made `matplotlib` required dependency (made dev easier to manage)

### Fix 

- Modified exposed utils

## v1.0.7 (2024-07-25)

### Feat

- Add support for loading JMP models into Python using JENN 

### Fix 

- Change default activation in `Parameters` class from `relu` to `tanh`
- Fix initialization of `sigma_x` and `sigma_y` to use `np.ones` (erroneously, it previously used `np.eye`)

### Docs

- Deleted `theory.pdf` (no longer needed now that paper is on ArXiv)
- Updated CONTRIBUTING to reflect `pixi` process (more simple)
- Added section about loading JMP models into JENN (with examples)

### Build

- Switched from `doit` to `pixi` (no need for a base environment anymore, more simple overall)
- Update GitHub Actions workflow to use `pixi` 

### Test 

- Added `nbmake` to test example notebooks during `qa` 
- Added unit tests for new JMP feature

## v1.0.6 (2024-06-18)

### Docs 

- Added link to technical paper on ArXiv (preprint) in README and `docs\index.rst`
- Fixed notation inconsistency in Jacobian matrix (data structures section)
- Updated `demo_4_rosenbrock.ipynb` with plot annotations (and fixed random seed)

### Refactor

- Switched order of indices `r` and `s` in `propagation.py` to match paper

## v1.0.5 (2024-05-11)

### Fix 

- missing dependencies (`jsonschema`, `jsonpointer`) 
- missing data (*.json was not being included in build, so added MANIFEST.in)
- typing oversight for python 3.8 (in `cost.py` and `sythetic.py`) 

## v1.0.4 (2024-05-08)

### Fix 

- Fixed random seed not working (previously not being passed to parameter initialization)
- Fixed `minibatch` issue throwing error below when `shuffle=False` and more than one batch
```
Traceback (most recent call last):
  File "C:\[...]\jenn\model.py", line 141, in fit
    self.history = train_model(
  File "C:\[...]\jenn\core\training.py", line 121, in train_model
    batches = data.mini_batches(batch_size, shuffle, random_state)
  File "C:\[...]\jenn\core\data.py", line 229, in mini_batches
    batches = mini_batches(X, batch_size, shuffle, random_state)
  File "C:\[...]\jenn\core\data.py", line 51, in mini_batches
    if mini_batch:
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

### Refactor

- Added jsonschema to validate reloaded parameters and check array shapes
- Added levels as input to `plot.contours`

### Features 

- Added optional ability to prioritize individual training points (useful to ensure more accuracy in known regions of interest)
- Added optional ability to warmstart; i.e. continue training from current parameters (without initialization)
- Exposed more hyperparameters pertaining to optimizer (e.g. tolerance stopping criteria) 
- Added option to use finite difference for generating synthetic data partials (used to study effect noisy partials)

### Documentation 

- Added airfoil notebook as example of large dataset
- Added surrogate-based optimization notebook to demonstrate benefit of JENN
- Updated theory.pdf

## v1.0.3 (2024-02-28)

### Fix 

- Updated annotations in `jenn.utils.plot` which were incompatible with Python 3.8 (causing runtime errors)
- Manually updated `__version__` number inside `__init__` (previous oversight) 

### Documentation 

- Update demo examples to use `from jenn.utils import plot` instead of `jenn.utils.plot` (which failed a test on Python 3.11.7)

## v1.0.2 (2024-02-25)

### Feature 

- Added support for `python >= 3.8` 

## v1.0.1 (2024-02-24)

### Documentation

- Update [paper link](https://github.com/shb84/JENN/blob/master/docs/theory.pdf) to point to version on `master` instead of `refactor` branch 

### Fix

- Fixed `minibatch` which was previously not reusing parameters from one batch to another 
- Fixed `random_state` which was previously was not being passed everywhere it should 
- Resolved rrror messages when optional `matplotlib` library not installed 

### Feature 

- Added support for `python >= 3.9` 

## v1.0.0 (2024-02-19)

_This release introduces breaking changes but makes the algorithm about 5x faster._ 

### Feature

- Added static, sensitivity profiles as new plotting utility
- Added `synthetic` module to create example data from canonical test functions 
- Added `evaluate` method to `NeuralNet` model (which does `predict` and `predict_partials` in one step more efficiently)

### Documentation

- Added documentation using `sphinx` and published on GitHub Pages
- Added more example notebooks

### Refactor 

- Moved core API into its own subpackage 
- Moved core API data management functionality into their own classes: `Parameters`, `DataSet`, `Cache` 
- Moved plotting module and metrics into utilities subpackage 
- Renamed core API modules, classes, and functions 
- Renamed user API modules, classes, and functions
- Changed user API by adding NeuralNet model
- Changed user API plotting utility functions names and kwards 
- Changed datastructure exposed to user (compared to `v0.1.0`, `X, Y, J` are now transposed)
- Simplified almost all functions for easier maintainability/readability

### Style

- Using `ruff`, `docformatter`, `black` and `mypy` for linting 

### Performance 

- Arrays are now updated in place (code about 5x faster)

### Tests

- Added unit tests using `pytest` 

## v0.1.0 (2021-03-30)

- First release of `jenn`