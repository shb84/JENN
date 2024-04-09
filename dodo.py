"""Define 'doit' tasks"""
import os
import sys 
import shutil
import doit 
import platform
import subprocess
from pathlib import Path
from packaging import version 

PYTHON_VERSION = version.parse(sys.version[:3]) 

if version.parse("3.10") <= PYTHON_VERSION:
    import tomllib  # type: ignore 
else: 
    from pip._vendor import tomli as tomllib


DOIT_CONFIG = {
    "backend": "sqlite3",
    "par_type": "thread",
    "default_tasks": [],
    "verbosity": 2,
}


def task_lock():
    """Re-generate lockfiles in deploy/conda/locks."""
    
    lockfile = f"{P.LOCKS}"
    specfile = f"{P.SPECS}"

    args = " ".join([f"-p {platform}" for platform in C.PLATFORMS]).split()
    args += ["--file", specfile]
    args += ["--lockfile", lockfile]

    env_info = (
        f"""{lockfile} {F.OKBLUE} {specfile}"""
    )

    def solve(): 
        solve_rc = subprocess.call(["conda-lock", *map(str, args)], cwd=str(P.ROOT))
        return solve_rc == 0

    return dict(
        file_dep=[specfile],
        actions=[
            lambda: U.printc(f"{F.LOCK}{env_info}", F.HEADER),
            solve,
            lambda: U.printc(f"{F.OK}{env_info}", F.OKGREEN),
        ],
        targets=[lockfile],
    )


def task_env():
    """Ensure conda environments."""

    prefix = P.PREFIX
    lockfile = P.LOCKS 
    history = prefix / C.HISTORY

    return dict(
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


def task_install():
    """Install locally."""

    return dict(
        **U.run(
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
            ok=OK.INSTALL,
        ),
    )


def task_lab():
    """Run JupyterLab (not run by default)."""

    def lab():
        proc = subprocess.Popen(
            [*U.run_args(), "jupyter", "lab", "--no-browser"]
        )
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.terminate()
            proc.wait()

    return dict(
        uptodate=[lambda: False],
        actions=[doit.tools.PythonInteractiveAction(lab)],
    )


def task_test():
    """Run test suite."""

    REPORT_PYTEST = P.REPORTS / "pytest" / U.this_platform()
    REPORT_PYTEST_COV_HTML = REPORT_PYTEST / "coverage" / "index.html"
    REPORT_PYTEST_COV_XML = REPORT_PYTEST / "coverage.xml"
    REPORT_PYTEST_HTML = REPORT_PYTEST / "report.html"
    REPORT_PYTEST_JUNIT = REPORT_PYTEST / "junit.xml"

    pytest_args = [
        "-vv",
        "--failed-first",
        f"--html={REPORT_PYTEST_HTML}",
        "--self-contained-html",
        f"--cov={C.PPT_DATA['project']['name']}",
        "--cov-context=test",
        f"--cov-report=html:{REPORT_PYTEST_COV_HTML.parent}",
        f"--cov-report=xml:{REPORT_PYTEST_COV_XML}",
        "--cov-report=term-missing:skip-covered",
        "-o=junit_family=xunit2",
        f"--junitxml={REPORT_PYTEST_JUNIT}",
    ]

    return dict(
        uptodate=[doit.tools.config_changed({"args": pytest_args})],
        **U.run(
            actions=[["pytest", *pytest_args]],
            targets=[
                REPORT_PYTEST_HTML,
                REPORT_PYTEST_COV_XML,
                REPORT_PYTEST_COV_HTML,
            ],
            ok=OK.PYTEST,
        ),
    )


def task_fix(): 
    """Apply automated code formatting."""
    
    yield dict(
        name="black",
        doc="aggressively format python code",
        **U.run(
            actions=[["black", "--quiet", *P.ALL_PY]],
            file_dep=[*P.ALL_PY, P.PPT],
            ok=OK.BLACKENED,
        ),
    )
    
    yield dict(
        name="ruff",
        doc="normalize python",
        **U.run(
            actions=[["ruff", "--fix-only", *P.ALL_PY]],
            file_dep=[*P.ALL_PY, P.PPT],
            ok=OK.RUFFENED,
        ),
    )

    yield dict(
        name="docformatter",
        doc="normalize python docstrings",
        **U.run(
            actions=[
                ["docformatter", "--in-place", *P.ALL_PY],
            ],
            file_dep=[*P.ALL_PY],
            ok=OK.DOCFORMATTED,
        ),
    ) 
 


def task_lint():
    """Check source code for style compliance."""    
    yield dict(
        name="black",
        doc="check python code for blackness",
        **U.run(
            actions=[["black", "--quiet", *P.ALL_PY]], 
            file_dep=[*P.ALL_PY, P.PPT],
        ),
    )

    yield dict(
        name="docformatter",
        doc="check python docstrings for fixable style issues",
        **U.run(
            actions=[["docformatter", "--check", *P.ALL_PY]],
            file_dep=[*P.ALL_PY],
        ),
    )

    yield dict(
        name="ruff",
        doc="check python code",
        **U.run(
            actions=[["ruff", "check", *P.ALL_PY]], 
            file_dep=[P.PPT, *P.ALL_PY]
        ),
    )

    REPORT_MYPY = P.REPORTS / "mypy" 
    REPORT_MYPY_HTML_INDEX = REPORT_MYPY / "index.html"
    REPORT_MYPY_TXT = REPORT_MYPY / "report.txt"
    REPORT_MYPY_QUALITY = REPORT_MYPY / "gitlab-code-quality.json"

    mypy_args = [
        f"--html-report={REPORT_MYPY}",
    ]
    
    yield dict(
        name="mypy",
        doc="check for well-typed python",
        **U.run(
            actions=[["mypy", *P.ALL_PY, *mypy_args]], 
            file_dep=[*P.ALL_PY, P.PPT],
            targets=[
                REPORT_MYPY_HTML_INDEX, 
                REPORT_MYPY_TXT, 
                REPORT_MYPY_QUALITY,
            ],
        ),
    )


def task_build():
    """Build wheel and *.tar.gz files."""

    yield dict(
        name="docs",
        uptodate=[lambda: False],
        **U.run(
            actions=[[
                "sphinx-build",
                "-b", 
                "html", 
                P.DOCS_SOURCE, 
                P.DOCS_BUILD / "html",
            ]],
            ok=OK.DOCS,
        ),
    )

    yield dict(
        name="dist",
        **U.run(
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


def task_release():
    """Release to TestPyPi only (PyPi is reserved for CI)."""
            
    yield dict(
        name=f"{C.PPT_DATA['project']['name']}-test-release",
        **U.run(
            [
                [
                    "python",
                    "-m",
                    "twine",
                    "upload",
                    P.DIST / "*", 
                    "--verbose",
                    "--skip-existing",
                    "--repository",
                    "testpypi",  # replace with pypi for actual upload 
                ]
            ],
            cwd=P.ROOT,
            ok=OK.RELEASE,
            file_dep=[OK.BUILD],
        )
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


#####################
# SUPPORT FUNCTIONS #
#####################


class P:
    """Paths."""
    DODO = Path(__file__)
    ROOT = DODO.parent
    BUILD = ROOT / "build"
    DOCS = ROOT / "docs"
    DIST = BUILD / "dist"
    SOURCE = ROOT / "src"
    DEPLOY = ROOT / "deploy"
    DEPLOY_SPECS = DEPLOY / "specs"
    DOCS_SOURCE = DOCS / "source"
    DOCS_BUILD = BUILD / "docs"
    PREFIX = ROOT / ".venv"
    LOCKS = ROOT / f"conda-lock.yml"
    SPECS = ROOT / "environment.yml"
    PPT = ROOT / "pyproject.toml"
    REPORTS = BUILD / "reports"
    ALL_PY = sorted(list(SOURCE.rglob('*.py')))
    EXAMPLES = DOCS / "examples"
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
    PPT_DATA = tomllib.loads(P.PPT.read_text(encoding="utf-8"))


class OK:
    """Success files for non-deterministic logs."""

    INSTALL = P.BUILD / "install.ok"
    PYTEST = P.BUILD / "test.ok"
    RUFFENED = P.BUILD / "lint.ruff.ok"
    BLACKENED = P.BUILD / "lint.black.ok"
    ISORTED = P.BUILD / "lint.isorted.ok"
    DOCFORMATTED = P.BUILD / "lint.docformatted.ok"
    DOCS = P.BUILD / "docs.ok"
    BUILD = P.BUILD / "build.ok"
    RELEASE = P.BUILD / "release.ok"


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
    def run_args(cls):
        """Execute a set of actions from within conda environment."""
        return [
            C.CONDA_EXE,
            "run",
            "--prefix",
            P.PREFIX,
            "--live-stream",
            "--no-capture-output",
        ]
    
    @classmethod
    def cmd(cls, *args, **kwargs):  # noqa: D102
        if "shell" not in kwargs:
            kwargs["shell"] = False
        return doit.tools.CmdAction(*args, **kwargs)

    @classmethod
    def run(cls, actions, ok=None, **kwargs):
        """Wrap run_args with pydoit structure."""
        run_args = U.run_args()
        file_dep = kwargs.pop("file_dep", [])
        targets = kwargs.pop("targets", [])
        task = dict(
            file_dep=file_dep,
            actions=[U.cmd([*run_args, *action], **kwargs) for action in actions],
            targets=[*targets],
        )

        if ok:
            task["actions"] = [U._unokit(ok), *task["actions"], *U._okit(ok)]
            task["targets"] += [ok]

        return task

    