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

## v2.0.1 (2026-07-17)

### Fix

- Fixed off-by-one in `data.py` mini-batching that dropped the last mini-batch
- Fixed `_safe_divide` in `data.py` mutating the caller's array (now uses `np.where`)
- Fixed gradient-enhancement weight in `cost.py` (applied as `gamma`, not `gamma^3`)
- Fixed backtracking line-search decay in `optimization.py` (now linear `tau`, not `tau^(2^i)`)
- Fixed ADAM bias counter in `optimization.py` to use object identity instead of `id()`
- Fixed invalid f-string format spec in `optimization.py` (`{y::.6f}` -> `{y:.6f}`)
- Updated `notebooks/runtime.ipynb` to the current API (`jenn.utilities.sample`
  with `jenn.synthetic_data.rastrigin`); it still called the pre-`v2.0.0`
  `jenn.synthetic.Rastrigin.sample`, breaking the notebook test suite

### Perf

- Vectorized the `n_x` partial-derivative loop in `next_layer_partials`
  (`propagation.py`) with a single `np.tensordot`, also removing a matmul that
  was computed twice per iteration
- Vectorized the `n_x` loop in `gradient_enhancement` (`propagation.py`) into
  BLAS-backed `dot`/`tensordot` contractions
- Replaced the `n_y`/`n_x` Python loops in `SquaredLoss.evaluate` and
  `GradientEnhancement.evaluate` (`cost.py`) with `np.sum(np.square(...))`
- Write the input-layer identity partial in place via broadcasting in
  `first_layer_partials` (`propagation.py`), removing the per-pass
  `(n_x, n_x, m)` allocation (and the now-unused `eye` helper)
- Compute `g'(z) * dA` once into a reused cache buffer in `next_layer_backward`
  (`propagation.py`, `cache.py`) instead of three times
- Made `Tanh`/`Relu` `first_derivative` fully in-place (`activation.py`),
  removing per-call temporaries

### Test

- Fixed `test_model_forward` to call `model_forward` (was erroneously calling `partials_forward`)
- Added a finite-difference gradient check for gradient-enhanced backprop with
  `n_x >= 2` (`test_propagation.py`), guarding the vectorized `n_x` paths
- Added `scripts/benchmark.py` and a `benchmark` pixi task to measure the
  training hot path (micro + airfoil end-to-end)

### Build

- Split monolithic GitHub Actions workflow into separate `ci`, `docs`, and `release` workflows
- Resolved CI lint failures from `ruff` 0.15 rule changes
- Added `[tool.docformatter]` config (`wrap-summaries = 0`) to stop RST module-docstring headers from being collapsed into the summary line
- Fixed multi-line `pixi` task commands (`test-unit`, `sphinx`, `fix-toml`) whose arguments were being split into separate shell commands by newlines
- Hardened the `release` workflow: gate on the full CI matrix (via `ci.yml`
  `workflow_call`) and a tag/`__version__` consistency check before any publish
- Added a local-wheel smoke test (install + import in a clean venv) in `build-dist`,
  plus a TestPyPI round-trip install/import gate before the PyPI publish
- Switched PyPI/TestPyPI publishing to Trusted Publishing (OIDC), removing stored
  API tokens
- Release notes are now populated automatically from `CHANGELOG.md` instead of
  being empty

## v2.0.0 (2025-12-06)

### Feat 

- Added prediction error histogram

### Fix 

- Fixed bug with line search (previously stalling or getting worse)
- Fixed bug with residuals-by-predicted (x-axis was not showing predicted correctly)
- Fixed bug with sensitivity profiler (index error causing it to fail for multiple responses)

### Test 

- Updated unit tests to reflect refactoring changes 
- Added testing for all supported Python versions (leveraging pixi)

### Build

- Added `pixi.toml` (rather than putting everything in `pyproject.toml`)
- Cleanup CI and updated worflow to test all supported python versions using pixi 

### Docs

- Simplified CONTRIBUTING 
- Updated README to reflect refactoring changes 
- Updated DOCS to reflect refactoring changes

### Style 

- Updated linting and annotations 

### Refactor

- Dropped support for Python 3.8 (because SMT no longer supports it and it doesn't handle annotations as well)
- Changed API by moving module `model.py` into `core` 
- Changed API by moving module `synthetic.py` into `synthetic_data` (synthetic functions are now modules not classes)
- Changed API by moving module `plot` into `post_processing` 
  - _Instead of `jenn.utils.plot.something()` it is now `jenn.plot_something()`_
  - _Modified signature and tweaked almost all plotting functions (adjusted notebooks accordingly)_
  - _Added plotting function to display histogram of prediction error (and added it to goodness of fit summary plots)_
- Changed name of `utils` to `utilities` and added `_sample.py` and `_finite_difference.py` modules
- Converted `load` method of `NeuralNet` to a classmethod to make more intuitive:
  - _Old pattern: `reloaded = NeuralNet(layer_sizes=[1, 2, 3]).load("save_params.json")`_
  - _New pattern: `reloaded = NeuralNet.load("save_params.json")`_
- Converted `load` method of `Parameters` to a classmethod to make more intuitive:
  - _Old pattern: `reloaded = Parameters(layer_sizes=[1, 2, 3]).load("save_params.json")`_
  - _New pattern: `reloaded = Parameters.load("save_params.json")`_
- Replaced `NeuralNet.evaluate(x)` by `NeuralNet.__call__(x)`
 
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