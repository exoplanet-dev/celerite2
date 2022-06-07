import nox

ALL_PYTHON_VS = ["3.8", "3.9", "3.10"]
TEST_CMD = ["coverage", "run", "-m", "pytest", "-v"]


@nox.session(python=ALL_PYTHON_VS)
def core(session):
    session.install(".[test]")
    session.run(*TEST_CMD, "python/test", *session.posargs)


@nox.session(python=ALL_PYTHON_VS)
def jax(session):
    session.install(".[test,jax]")
    session.run(*TEST_CMD, "python/test/jax", *session.posargs)


@nox.session(python=ALL_PYTHON_VS)
def pymc3(session):
    session.install(".[test,pymc3]")
    session.run(*TEST_CMD, "python/test/pymc3", *session.posargs)


@nox.session(python=ALL_PYTHON_VS)
def pymc4(session):
    session.install(".[test,pymc4]")
    session.run(*TEST_CMD, "python/test/pymc4", *session.posargs)


@nox.session(python=ALL_PYTHON_VS)
def full(session):
    session.install(".[test,jax,pymc3,pymc4]")
    session.run(*TEST_CMD, "python/test", *session.posargs)


@nox.session
def lint(session):
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)
