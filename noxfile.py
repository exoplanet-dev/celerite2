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
def pymc4(session):
    session.install(".[test,pymc4]")
    _session_run(session, "python/test/pymc4")


@nox.session(python=ALL_PYTHON_VS)
def pymc4_jax(session):
    session.install(".[test,jax,pymc4]")
    _session_run(session, "python/test/pymc4/test_pymc4_ops.py")


@nox.session(python=ALL_PYTHON_VS)
def full(session):
    session.install(".[test,jax,pymc3,pymc4]")
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
