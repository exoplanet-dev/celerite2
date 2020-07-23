def pytest_configure(config):
    try:
        import theano
    except ImportError:
        return

    import platform

    theano.config.floatX = "float64"
    if platform.system() == "Darwin":
        theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
    config.addinivalue_line("filterwarnings", "ignore")
