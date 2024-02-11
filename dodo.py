"""Define 'doit' tasks"""
import os
import shutil
import doit 
import platform
import subprocess
import tomllib
from collections.abc import Iterable
from typing import Any
from pathlib import Path
from typing_extensions import TypedDict


DOIT_CONFIG = {
    "backend": "sqlite3",
    "par_type": "thread",
    "default_tasks": [],
    "verbosity": 2,
}

     
#########
# TASKS #
#########


def task_lock():
    """Re-generate lockfiles in deploy/conda/locks."""
    
    for platform in C.PLATFORMS: 
        for env_stem, env_specs in P.SPECS.items(): 
            yield from U.lock(
                    env_stem,
                    platform,
                    env_specs,
                )


def task_env():
    """Ensure conda environments."""
    for env_name in P.SPECS:
        yield from U.env(env_name)


def task_package():
    """Build wheel and *.tar.gz files."""
    yield dict(
        name=f"{C.PPT_DATA['project']['name']}-build",
        **U.run_in(
            "ci",
            [
                [
                    "python",
                    "-m",
                    "build",
                    "--outdir",
                    P.DIST,
                ]
            ],
            cwd=P.ROOT,
            ok=OK.BUILD,
        ),
    )


def task_install():
    """Install locally."""
    yield dict(
        name=C.PPT_DATA['project']['name'],
        **U.run_in(
            "ci",
            [
                [
                    "pip",
                    "install",
                    "-e",
                    ".",
                    "--no-deps",
                    "--ignore-installed",
                ]
            ],
            cwd=P.ROOT,
            ok=OK.INSTALL,
        ),
    )


def task_notebooks():
    """Execute all notebooks (from command line)."""

    def execute():
        _, run_args = U.run_args("ci")
        proc = subprocess.Popen(
            [*run_args, "jupyter", "execute", *P.DEMO_NOTEBOOKS]
        )
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.terminate()
            proc.wait()

    yield dict(
        name="notebooks",
        uptodate=[lambda: False],
        file_dep=[OK.INSTALL],
        actions=[doit.tools.PythonInteractiveAction(execute)],
    )


def task_lab():
    """Run JupyterLab."""

    def lab():
        _, run_args = U.run_args("ci")
        proc = subprocess.Popen(
            [*run_args, "jupyter", "lab", "--no-browser"]
        )
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.terminate()
            proc.wait()

    yield dict(
        name="lab",
        uptodate=[lambda: False],
        file_dep=[OK.INSTALL],
        actions=[doit.tools.PythonInteractiveAction(lab)],
    )


def task_test():
    """Run test suite."""

    PYTEST = P.REPORTS / "pytest" / U.this_platform()
    PYTEST_COV_HTML = PYTEST / "coverage" / "index.html"
    PYTEST_COV_XML = PYTEST / "coverage.xml"
    PYTEST_HTML = PYTEST / "report.html"

    pytest_args = [
        "-vv",
        "--failed-first",
        f"--html={PYTEST_HTML}",
        "--self-contained-html",
        f"--cov={C.PPT_DATA['project']['name']}",
        "--cov-context=test",
        f"--cov-report=html:{PYTEST_COV_HTML.parent}",
        f"--cov-report=xml:{PYTEST_COV_XML}",
        "--cov-report=term-missing:skip-covered",
    ]

    yield dict(
        name="pytest",
        uptodate=[doit.tools.config_changed({"args": pytest_args})],
        **U.run_in(
            "ci",
            [["pytest", *pytest_args]],
            file_dep=[
                OK.INSTALL, 
                P.PPT,
            ],
            targets=[
                PYTEST_HTML,
                PYTEST_COV_XML,
                PYTEST_COV_HTML,
            ],
        ),
    )


def task_docs():
    """Update sphinx documentation."""

    yield dict(
        name="sphinx",
        doc="create docs",
        uptodate=[lambda: False],
        **U.run_in(
            "ci",
            actions=[[
                # "sphinx-build",
                # "-b", 
                # "html", 
                "sphinx-multiversion",
                P.DOCS_SOURCE, 
                P.DOCS_BUILD / "html",
            ]],
            ok=OK.DOCS,
            file_dep=[OK.INSTALL],
        ),
    )



def task_fix(): 
    """Apply automated code formatting."""
    
    yield dict(
        name="black",
        doc="aggressively format python code",
        **U.run_in(
            "qa",
            actions=[["black", "--quiet", *P.ALL_PY]],
            file_dep=[*P.ALL_PY, P.PPT],
            ok=OK.BLACKENED,
        ),
    )
    
    yield dict(
        name="ruff",
        doc="normalize python",
        **U.run_in(
            env_="qa",
            actions=[["ruff", "--fix-only", *P.ALL_PY]],
            file_dep=[*P.ALL_PY, OK.BLACKENED, P.PPT],
            ok=OK.RUFFENED,
        ),
    )

    yield dict(
        name="docformatter",
        doc="normalize python docstrings",
        **U.run_in(
            "qa",
            actions=[
                ["docformatter", "--in-place", *P.ALL_PY],
            ],
            file_dep=[*P.ALL_PY, OK.RUFFENED],
            ok=OK.DOCFORMATTED,
        ),
    ) 


def task_lint():
    """Check source code for style compliance."""    
    yield dict(
        name="black",
        doc="check python code for blackness",
        **U.run_in(
            "qa", actions=[["black", "--quiet", *P.ALL_PY]], file_dep=[*P.ALL_PY, P.PPT],
        ),
    )

    yield dict(
        name="docformatter",
        doc="check python docstrings for fixable style issues",
        **U.run_in(
            "qa",
            actions=[["docformatter", "--check", *P.ALL_PY]],
            file_dep=[*P.ALL_PY],
        ),
    )

    yield dict(
        name="ruff",
        doc="check python code",
        **U.run_in(
            "qa", actions=[["ruff", *P.ALL_PY]], file_dep=[P.PPT, *P.ALL_PY]
        ),
    )

    yield dict(
        name="mypy",
        doc="check for well-typed python",
        **U.run_in(
            "qa", actions=[["mypy", *P.ALL_PY]], file_dep=[*P.ALL_PY, P.PPT],
        ),
    )


#####################
# SUPPORT FUNCTIONS #
#####################

class TUTF8(TypedDict):
    """A type for encoding."""

    encoding: str


class P:
    """Paths."""
    DODO = Path(__file__)
    ROOT = DODO.parent
    BUILD = ROOT / "build"
    DOCS = ROOT / "docs"
    EXAMPLES = DOCS / "examples"
    DIST = ROOT / "./build/dist"
    SOURCE = ROOT / "src"
    DEPLOY = ROOT / "deploy"
    DEPLOY_SPECS = DEPLOY / "specs"
    DOCS_SOURCE = DOCS / "source"
    DOCS_BUILD = BUILD / "docs"
    ENVS = ROOT / ".envs"
    LOCKS = DEPLOY / "locks"
    SPECS = {
        "ci": [
            DEPLOY_SPECS / "base.yml", 
            DEPLOY_SPECS / "run.yml",
            DEPLOY_SPECS / "lab.yml",
            DEPLOY_SPECS / "docs.yml",
            DEPLOY_SPECS / "test.yml",
        ],
        "qa": [
            DEPLOY_SPECS / "base.yml", 
            DEPLOY_SPECS / "qa.yml",
        ],
    }
    PPT = ROOT / "pyproject.toml"
    REPORTS = BUILD / "reports"
    ALL_PY = sorted(list(SOURCE.rglob('*.py')))
    DEMO_NOTEBOOKS = [
        item 
        for item in sorted(list(EXAMPLES.rglob('*.ipynb')))
        if not ".ipynb_checkpoints" in str(item)
    ]



class B: 
    """Booleans."""
    IS_WIN = platform.system() == "Windows"
    IS_MAC = platform.system() == "Darwin"
    IS_LINUX = not (IS_WIN or IS_MAC)
    HAS_COLOR = True
    if IS_WIN:
        try:
            import colorama
            colorama.init()
            HAS_COLOR = True
        except ImportError:
            HAS_COLOR = False 


class F: 
    """Formatting."""
    HEADER = "\033[95m" if B.HAS_COLOR else "=== "
    OKBLUE = "\033[94m" if B.HAS_COLOR else "+++ "
    OKGREEN = "\033[92m" if B.HAS_COLOR else "*** "
    WARNING = "\033[93m" if B.HAS_COLOR else "!!! "
    FAIL = "\033[91m" if B.HAS_COLOR else "XXX "
    ENDC = "\033[0m" if B.HAS_COLOR else ""
    BOLD = "\033[1m" if B.HAS_COLOR else "+++ "
    UNDERLINE = "\033[4m" if B.HAS_COLOR else "___ "
    BOOM = "XX " if B.IS_WIN else "💥 "
    LOCK = "LOCK " if B.IS_WIN else "🔒 "
    OK = "OK " if B.IS_WIN else "👌 "
    PLAY = "DO " if B.IS_WIN else "▶️ "
    SKIP = "SKIP " if B.IS_WIN else "⏭️ "
    STAR = "YAY " if B.IS_WIN else "🌟 "
    UPDATE = "UPDATE " if B.IS_WIN else "🔄 "
    UTF8 = TUTF8(encoding="utf-8")


class C: 
    """Constants."""
    WIN_PLATFORM = 'win-64'
    LINUX_PLATFORM = 'linux-64'
    OSXARM64_PLATFORM = 'osx-arm64'
    OSX64_PLATFORM = 'osx-64'
    LINUX_PLATFORMS = [
        LINUX_PLATFORM, 
        OSX64_PLATFORM, 
        OSXARM64_PLATFORM,
    ]
    PLATFORMS = [
        WIN_PLATFORM, 
        LINUX_PLATFORM, 
        OSX64_PLATFORM, 
        OSXARM64_PLATFORM,
    ]
    CONDA_EXE = os.environ.get(
        "CONDA_EXE", shutil.which("conda") or shutil.which("conda.exe")
    )
    MAMBA_EXE = os.environ.get(
        "MAMBA_EXE", shutil.which("mamba") or shutil.which("mamba.exe")
    )
    HISTORY = "conda-meta/history"
    PPT_DATA = tomllib.loads(P.PPT.read_text(**F.UTF8))


class OK:
    """Success files for non-deterministic logs.

    Note: envs are determined via conda-meta/history file
    """

    INSTALL = P.BUILD / "install.ok"
    RUFFENED = P.BUILD / "lint.ruff.ok"
    BLACKENED = P.BUILD / "lint.black.ok"
    DOCFORMATTED = P.BUILD / "lint.docformatted.ok"
    DOCS = P.BUILD / "docs.ok"
    PYTEST = P.BUILD / "pytest.ok"
    BUILD = P.BUILD / "build.ok"


class U: 

    @classmethod
    def _unokit(cls, ok):
        def action():
            if ok.exists():
                ok.unlink()
            return True

        return action

    @classmethod
    def _okit(cls, ok):
        def action():
            ok.write_text("OK")
            return True

        return [(doit.tools.create_folder, [ok.parent]), action]
        
    @classmethod
    def printc(cls, statement: str, level: str = F.OKBLUE) -> None:
        """Print nicely with color."""
        print(f"{level}{statement}{F.ENDC}", flush=True)
    
    @classmethod
    def lock(
        cls, env_stem: str, platform: str, recipes: list[Path] | None = None
    ):
        """Generate a lock for recipe(s)."""
        env_files = recipes or []
        args = ["--platform", platform]
        stem = f"{env_stem}-{platform}"
        lockfile = P.LOCKS / f"{stem}.conda.lock"

        specs: list[Path] = []

        for env_file in sorted(env_files):
            candidates = [f"{env_file.stem}", f"{env_file.stem}-{platform}"]
            if platform in C.LINUX_PLATFORMS:
                candidates += [f"{env_file.stem}-unix"]
            for fname in candidates:
                spec = env_file.parent / f"{fname}.yml"
                if spec.exists():
                    specs += [spec]

        specs = sorted(set(specs))

        args += sum([["--file", spec] for spec in specs], [])
        args += [
            # "--filename-template",
            "--lockfile",
            env_stem + f"-{platform}.conda.lock",
        ]
        env_info = (
            f"""{lockfile.name.rjust(30)}  """
            f"""{F.OKBLUE}{"  ".join([s.stem for s in specs])}"""
        )
        yield dict(
            name=f"""{env_stem}:{platform}""",
            file_dep=specs,
            actions=[
                (doit.tools.create_folder, [P.LOCKS]),
                lambda: cls.printc(f"{F.LOCK}{env_info}", F.HEADER),
                (U.solve, [args]),
                lambda: cls.printc(f"{F.OK}{env_info}", F.OKGREEN),
            ],
            targets=[lockfile],
        )

    @classmethod
    def this_platform(cls): 
        if platform.system() == "Darwin": 
            if platform.machine() == "arm64": 
                return C.OSXARM64_PLATFORM
            else: 
                return C.OSX64_PLATFORM
        elif platform.system() == "Windows": 
            return C.WIN_PLATFORM
        elif platform.system() == "Linux": 
            return C.LINUX_PLATFORM
    
    @classmethod
    def solve(cls, args: Iterable[Any]):
        """Create the lock file."""
        solve_rc = 1
        base_args = [
            "conda-lock", 
            # "--kind=explicit"  
        ]

        for solver_args in [["--micromamba"], ["--mamba"], ["--no-mamba"], []]:
            solve_rc = subprocess.call(
                [*base_args, *solver_args, *map(str, args)], cwd=str(P.LOCKS)
            )
            
            if solve_rc == 0:
                cls.printc(
                    f"Solved using {' '.join(solver_args) or 'default config.'}",
                    F.OKBLUE,
                )
                break

        return solve_rc == 0
    
    @classmethod
    def env(cls, name: str):
        """Create an environment from a lockfile."""
        prefix = P.ENVS / name
        lockfile = P.LOCKS / f"{name}-{cls.this_platform()}.conda.lock"
        history = prefix / C.HISTORY
        yield dict(
            name=name,
            file_dep=[lockfile],
            actions=[
                U.cmd(
                    [
                        "conda-lock",
                        "install",
                        "--prefix",
                        prefix,
                        lockfile,
                    ],
                ),
            ],
            targets=[history],
        )

    @classmethod
    def run_args(cls, env):
        """Execute a set of actions from within conda environment."""
        prefix = P.ENVS / env
        conda = C.MAMBA_EXE if C.MAMBA_EXE else C.CONDA_EXE
        run_args = [
            conda,
            "run",
            "--prefix",
            prefix,
            "--live-stream",
            "--no-capture-output",
        ]
        return prefix, run_args
    
    @classmethod
    def cmd(cls, *args, **kwargs):  # noqa: D102
        if "shell" not in kwargs:
            kwargs["shell"] = False
        return doit.tools.CmdAction(*args, **kwargs)

    @classmethod
    def run_in(cls, env_, actions, ok=None, **kwargs):
        """Wrap run_args with pydoit structure."""
        prefix, run_args = U.run_args(env_)
        history = prefix / "conda-meta/history"
        file_dep = kwargs.pop("file_dep", [])
        targets = kwargs.pop("targets", [])
        task = dict(
            file_dep=[history, *file_dep],
            # actions=[U.cmd([*run_args, *action], **kwargs) for action in actions],
            actions=[U.cmd([*run_args, *action], **kwargs) for action in actions],
            targets=[*targets],
        )

        if ok:
            task["actions"] = [U._unokit(ok), *task["actions"], *U._okit(ok)]
            task["targets"] += [ok]

        return task

    