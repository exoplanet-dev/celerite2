def pytest_configure(config):
    try:
        import theano
    except ImportError:
        return

    import platform

    theano.config.floatX = "float64"
    theano.config.compute_test_value = "raise"
    if platform.system() == "Darwin":
        theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
    config.addinivalue_line("filterwarnings", "ignore")
