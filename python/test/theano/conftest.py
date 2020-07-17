def pytest_configure(config):
    import platform

    import theano

    theano.config.floatX = "float64"
    if platform.system() == "Darwin":
        theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
    config.addinivalue_line("filterwarnings", "ignore")
