.. _MCP: https://modelcontextprotocol.io
.. _Claude Code: https://docs.anthropic.com/en/docs/claude-code

MCP Server
==========

JENN ships an optional `Model Context Protocol <MCP_>`_ server so an AI agent
(e.g. `Claude Code`_) can build and validate a surrogate model end to end — from
data (with optional partials) to a portable, reloadable artifact — without
writing a training script by hand. The agent owns architecture and
hyperparameter search; the tools are thin wrappers over ``jenn.NeuralNet``.

Installation
------------

The server needs the optional ``mcp`` extra and Python >= 3.10. There are two
ways to make it available to your agent; pick the one that matches how you work.

Path 1 — from a clone (contributors, easiest)
.............................................

If you have cloned the repository there is **nothing to install or register**.
The clone ships a checked-in ``.mcp.json`` that launches the server inside the
project's ``pixi`` ``dev`` environment (``pixi run -e dev jenn-mcp``), where
``jenn[mcp]`` is already installed. Just start your agent **from the project
root**::

    cd /path/to/JENN
    claude            # approve the "jenn" server once when prompted

The ``jenn`` tools are now available in that session. The catch: this works only
from *within the clone*, because the ``pixi`` wrapper resolves against the
project's manifest. Launch Claude Code from an unrelated directory and the
``jenn`` server will not start — that is what Path 2 is for.

Path 2 — anywhere (install once, use in any project)
....................................................

To use the server outside the clone — in any project on your machine — install
``jenn[mcp]`` so its ``jenn-mcp`` command lands on your ``PATH``, then register
it at **user scope**. The cleanest option is `pipx <https://pipx.pypa.io>`_: it
installs the package into its own isolated environment and exposes only the
``jenn-mcp`` command on your ``PATH`` (so it never touches your system Python).

**1. Install pipx** (once, if you do not already have it). On macOS::

    brew install pipx
    pipx ensurepath      # wires ~/.local/bin into your shell profile; open a new terminal

On Linux (or without Homebrew)::

    python3 -m pip install --user pipx
    python3 -m pipx ensurepath      # then open a new terminal

**2. Install the server** (requires Python >= 3.10)::

    pipx install "jenn[mcp]"
    which jenn-mcp       # sanity check: the command now resolves

**3. Register it at user scope** so Claude Code offers it in every project,
regardless of directory::

    claude mcp add --scope user --transport stdio jenn -- jenn-mcp

(Plain ``pip install "jenn[mcp]"`` works too, as long as the resulting
``jenn-mcp`` is on the ``PATH`` your agent sees; ``pipx`` just keeps it isolated
and easier to reason about. Later, ``pipx list`` shows where it lives and
``pipx uninstall jenn`` removes it cleanly.)

Why two paths? Whatever you configure, your agent runs *exactly* the command you
give it, as a subprocess — and that command must resolve to a Python that has
``jenn[mcp]`` installed. Path 1 guarantees this with the ``pixi run -e dev``
wrapper (but only inside the clone); Path 2 guarantees it by putting
``jenn-mcp`` on your ``PATH`` (so it works anywhere). To launch the server by
hand for testing, use ``jenn-mcp`` or ``python -m jenn.mcp``.

Setting ``JENN_DIR``
--------------------

The server can discover data and model files on disk and read or write files by
path. It scans a single **root directory**, chosen as ``$JENN_DIR`` if set,
otherwise a ``.jenn_dir`` folder in the directory the server was launched from
(created on first use). That default is deliberately a folder JENN owns: it
keeps discovery focused on your data instead of everything else a project
contains, and keeps exported models from scattering among your sources.

Point it at the folder holding your data and models *before* launching::

    JENN_DIR=/path/to/my/project jenn-mcp

or set ``JENN_DIR`` in the ``env`` block of your MCP client configuration. An
explicit ``JENN_DIR`` is taken as given and is *not* created for you, so a
mistyped path surfaces as an error rather than as an empty folder. The root
governs the ``jenn://files`` resources (below) and is the base the file tools
resolve relative paths against, so setting it well is what makes "just find my
data" work.

A convenient convention is a dedicated ``~/.jenn`` folder, set once in your
shell profile so every session and every server you launch sees it::

    mkdir -p ~/.jenn

Then persist ``JENN_DIR`` in your shell's startup file. For zsh (the macOS
default)::

    echo 'export JENN_DIR="$HOME/.jenn"' >> ~/.zshrc
    source ~/.zshrc

For bash, use ``~/.bashrc`` (Linux) or ``~/.bash_profile`` (macOS) in place of
``~/.zshrc``. Now drop your data and exported models under ``~/.jenn`` and the
server finds them automatically.

Tools, resource, and prompt
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Tool
     - Purpose
   * - ``ingest``
     - Load training data from a ``.csv`` or ``.npz`` file and register it as a
       reusable dataset; returns a ``dataset_id`` and reports which partials are
       present vs. missing.
   * - ``list_datasets``
     - List the datasets held in the current session (shapes, column names,
       partial availability).
   * - ``train``
     - Fit a surrogate from inline arrays or a ``dataset_id`` (plus optional
       partials); returns a ``model_id`` and training-set metrics.
   * - ``evaluate``
     - Score a model on held-out data (or its training data): response and
       per-partial R2, RMSE, and max error.
   * - ``export``
     - Save a model to JENN's native JSON, reloadable with
       ``jenn.NeuralNet.load``.
   * - ``list_models``
     - List the models held in the current session.
   * - ``load_model``
     - Reload a previously ``export``-ed model file into a new session so it can
       be reused.
   * - ``predict``
     - Run a trained or reloaded model on new inputs (inline or from a file);
       optionally returns the Jacobian and/or writes results to a file.

The ``jenn://files`` **resource** lists JENN-relevant files under ``JENN_DIR`` —
CSV/NPZ data files (with their columns / array names) and exported model JSONs —
so you can ``@``-mention it to discover files instead of pasting paths. Every one
of those files is *also* advertised individually as ``jenn://files/<name>``
(sub-folders included), so an agent's ``@`` menu lists them one by one and you
can pick the file you want rather than typing its path. Reading one attaches a
short metadata card — format, columns or array names, a model's architecture, the
absolute path, and a few preview lines for a CSV — not the file's rows: the data
itself stays server-side behind ``ingest``. The ``surrogate_workflow`` **prompt**
walks an agent through the full recipe.

Because a single training run is stochastic, the tools advise re-running with a
different seed and evaluating on held-out data before acting on a diagnosis.

Data orientation
----------------

Inline data crossing the tool boundary is **row-per-sample** (one row per
training point): ``x`` is ``(m, n_x)``, ``y`` is ``(m, n_y)``, ``dydx`` is
``(m, n_y, n_x)`` — the convention agents and tabular data most naturally
produce; the server transposes internally to JENN's feature-first core layout.

Files keep each format's **native** orientation: a ``.csv`` is a row-per-sample
table (what a spreadsheet exports), while a ``.npz`` is feature-first — ``x`` of
shape ``(n_x, m)`` — matching ``jenn.utilities.load_npz`` and the core arrays.
So an ``.npz`` you hand to ``predict`` (or receive from it) uses the core axis
order, the transpose of the inline arrays. Each channel matches the convention
its producer most naturally emits.

Tutorial: build a surrogate end to end
--------------------------------------

This walkthrough trains a surrogate of the 2-D Rastrigin function and reuses it
in a fresh session. The practice data is shipped with the docs at
``docs/examples/data/rastrigin.csv`` — 100 samples with columns
``x1, x2, y, slope_wrt_x1``. Note it carries the partial ``d(y)/d(x1)`` but
**not** ``d(y)/d(x2)`` — a realistic "some partials available" case that shows
how JENN gamma-masks the missing one.

First make sure the server will see the practice data. In the normal setup your
agent (e.g. Claude Code) launches the server for you, so the goal is simply to
get ``rastrigin.csv`` under the folder ``JENN_DIR`` points at.

The simplest way — if you followed the ``~/.jenn`` convention above — is to copy
the file there and launch your agent as usual::

    cp docs/examples/data/rastrigin.csv ~/.jenn/

Prefer a one-off instead of a permanent directory? Launch your agent with a
per-session ``JENN_DIR`` (an absolute path, so it does not depend on where the
server is started). Put the prefix on the **agent** command — it passes the
variable down to the server it spawns::

    JENN_DIR="$PWD/docs/examples/data" claude   # run from the project root

Only launching the server *by hand* (rather than through an agent)? Then the
prefix rides on ``jenn-mcp`` instead, run from the project root so the relative
path resolves::

    JENN_DIR=docs/examples/data jenn-mcp

With the data in place, confirm the ``jenn`` server is actually connected before
you start: run ``/mcp`` in Claude Code and check that ``jenn`` appears with its
tools (``ping``, ``ingest``, ``train`` …). As a smoke test, paste:

.. code-block:: text

    Run the jenn ping tool and tell me what it returns.

You should get ``pong``. If ``jenn`` is not listed, revisit the installation
paths above — for Path 1 the agent must be launched from the project root.

Then work through the steps below. Each gives a copy-paste **prompt** and the
tool it drives.

**1. Discover the data.** File discovery is exposed as an MCP *resource*, not a
tool, so you attach it explicitly with an ``@`` mention rather than a vague ask.
The mention syntax is ``@<server>:<uri>`` — here ``@jenn:jenn://files``:

.. code-block:: text

    List @jenn:jenn://files

That attaches the listing, which shows ``rastrigin.csv`` (with its columns) under
``JENN_DIR``. In Claude Code you can also type ``@`` and pick from the menu: the
files under ``JENN_DIR`` appear there individually, so mentioning one directly
works too and attaches just that file's card —

.. code-block:: text

    What columns does @jenn:jenn://files/rastrigin.csv have?

Without an ``@`` mention — e.g. a plain *"List the JENN files you can see"* — the
agent has no resource to read and will just search the filesystem instead.
(Discovery is optional: you can skip straight to ingesting the file by name in
step 2, since the tools read the file for you.)

**2. Ingest it as a dataset.**

.. code-block:: text

    Ingest rastrigin.csv with inputs x1 and x2, output y, and the derivative of y with respect to x1 in column slope_wrt_x1.

Drives ``ingest``. The reply includes a ``dataset_id`` and notes that
``d(y)/d(x2)`` is absent and will be gamma-masked to 0 at train time.

**3. Train a surrogate.**

.. code-block:: text

    Train a small JENN surrogate on that dataset and show me the fit.

Drives ``train``. Read ``training_metrics`` — the response R2, and the
per-partial R2 for the one available partial (the missing one is listed under
``partials.ignored``, not scored).

**4. Sanity-check with a second run.**

.. code-block:: text

    Re-train with a different random seed and compare — is the fit stable?

A single run is stochastic; comparing runs is the guarded way to judge the fit
before changing anything.

**5. Export the model.**

.. code-block:: text

    Save the model to rastrigin_model.json.

Drives ``export``. A bare filename is resolved under ``JENN_DIR`` (an absolute
path is used as-is), so the model lands right where the ``jenn://files`` resource
will find it — no full path needed.

**6. (New session) Reload and predict.**

Start a fresh agent session — the in-memory models are gone, but the file
persists. Reload it by the same bare name:

.. code-block:: text

    Load rastrigin_model.json, then predict y at x1=0.2, x2=-0.4.

Drives ``load_model`` then ``predict``. ``load_model`` also resolves a bare name
under ``JENN_DIR``, so no full path is needed. To also get slopes, ask:

.. code-block:: text

    Predict again and include the partial derivatives.

That closes the loop: train once, reuse later, no retraining and no pasted
arrays.

Where to go next
----------------

- The tools mirror ``jenn.NeuralNet`` — see :doc:`quickstart` for the Python API.
- For the math and array shapes, see the Data Structures section of
  :doc:`quickstart`.
