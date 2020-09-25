.. _install:

Installation
============

.. note:: *celerite2* requires Python 3.6 and later.

Using pip
---------

The best way to install *celerite2* is using `pip <https://pip.pypa.io>`_:

.. code-block:: bash

    python -m pip install -U celerite2


.. _source:

From Source
-----------

The source code for *celerite2* can be downloaded and installed `from GitHub
<https://github.com/exoplanet-dev/celerite2>`_ by running

.. code-block:: bash

    git clone --recursive https://github.com/exoplanet-dev/celerite2.git
    cd celerite2
    python -m pip install -e .


Testing
-------

To run the unit tests, install the development dependencies using pip:

.. code-block:: bash

    python -m pip install ".[test]"

and then execute:

.. code-block:: bash

    python -m pytest -v python/test

All of the tests should (of course) pass.
If any of the tests don't pass and if you can't sort out why, `open an issue
on GitHub <https://github.com/exoplanet-dev/celerite2/issues>`_.

To test the interfaces (for example, the Theano interface), run:

.. code-block:: bash

    python -m pip install ".[test,theano]"
    python -m pytest -v python/test/theano
