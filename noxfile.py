import nox

ALL_PYTHON_VS = ["3.8", "3.9", "3.10"]
TEST_CMD = ["coverage", "run", "-m", "pytest", "-v"]


def _session_run(session, path):
    if len(session.posargs):
        session.run(*TEST_CMD, *session.posargs)
    else:
        session.run(*TEST_CMD, path, *session.posargs)


@nox.session(python=ALL_PYTHON_VS)
def core(session):
    session.install(".[test]")
    _session_run(session, "python/test")


@nox.session(python=ALL_PYTHON_VS)
def jax(session):
    session.install(".[test,jax]")
    _session_run(session, "python/test/jax")


@nox.session(python=ALL_PYTHON_VS)
def pymc3(session):
    session.install(".[test,pymc3]")
    _session_run(session, "python/test/pymc3")


@nox.session(python=ALL_PYTHON_VS)
def pymc(session):
    session.install(".[test,pymc]")
    _session_run(session, "python/test/pymc")


@nox.session(python=ALL_PYTHON_VS, venv_backend="mamba")
def pymc_mamba(session):
    session.conda_install("pymc", channel="conda-forge")
    session.install(".[test,pymc]")
    _session_run(session, "python/test/pymc")


@nox.session(python=ALL_PYTHON_VS)
def pymc_jax(session):
    session.install(".[test,jax,pymc]")
    _session_run(session, "python/test/pymc/test_pymc_ops.py")


@nox.session(python=ALL_PYTHON_VS)
def full(session):
    session.install(".[test,jax,pymc3,pymc]")
    _session_run(session, "python/test")


@nox.session(python=ALL_PYTHON_VS, venv_backend="mamba")
def full_mamba(session):
    session.conda_install("jax", "pymc3", "pymc", channel="conda-forge")
    session.install(".[test]")
    _session_run(session, "python/test")


@nox.session
def lint(session):
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session(venv_backend="mamba")
def docs(session):
    import yaml

    with open("docs/environment.yml", "r") as f:
        env = yaml.safe_load(f)

    conda_deps = list(
        filter(lambda s: isinstance(s, str), env["dependencies"])
    )
    pip_deps = next(
        filter(lambda s: not isinstance(s, str), env["dependencies"])
    )["pip"]

    session.conda_install(*conda_deps, channel="conda-forge")
    session.install(*pip_deps)
    session.install(".")

    with session.chdir("docs"):
        session.run(
            *"python -m sphinx -T -E -b dirhtml -d _build/doctrees -D language=en . _build/html".split()
        )
