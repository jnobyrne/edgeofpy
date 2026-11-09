Installation
============

edgeofpy requires Python 3.10 or later. Install it from PyPI with:

.. code-block:: bash

   pip install edgeofpy

This also installs its dependencies: NumPy (>= 1.22), SciPy (>= 1.7),
Matplotlib (>= 3.5) and powerlaw (>= 1.5).

To upgrade to the latest version:

.. code-block:: bash

   pip install --upgrade edgeofpy

Development version
-------------------

To install the latest code from GitHub, with the test dependencies:

.. code-block:: bash

   git clone https://github.com/jnobyrne/edgeofpy.git
   cd edgeofpy
   pip install -e ".[test]"
   pytest
