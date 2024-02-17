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

<!--next-version-placeholder-->

## [Unreleased] - yyyy-mm-dd

_This release introduces breaking changes but makes the algorithm about 5x faster._ 

### Feature

- Added static, sensitivity profiles as new plotting utility
- Added `synthetic` module to create example data from canonical test functions 
- Added `evaluate` method to `NeuralNet` model (which does `predict` and `predict_partials` in one step more efficiently)

### Documentation

- Added documentation using `sphinx` and published on GitHub Pages

### Refactor 

- Moved core API into its own subpackage 
- Moved core API data management functionality into their own classes: `Parameters`, `DataSet`, `Cache` 
- Moved plotting module and metrics into utilities subpackage 
- Renamed core API modules, classes, and functions 
- Renamed user API modules, classes, and functions
- Changed user API by adding NeuralNet model
- Changed user API plotting utility functions names and kwards 
- Changed datastructure exposed to user (compared to `v0.1.01`, `X, Y, J` are now transposed)
- Simplified almost all functions for easier maintainability/readability

### Style

- Using `ruff`, `docformatter`, `black` and `mypy` for linting 

### Performance 

- Arrays updated in place (code about 5x faster)

### Tests

- Added unit tests using `pytest` 

## v0.1.0 (2021-03-30)

- First release of `jenn`