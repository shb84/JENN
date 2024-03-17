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

## v1.0.3.dev0 (2024-02-28)

### Refactor

- Added jsonschema to validate reloaded parameters

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