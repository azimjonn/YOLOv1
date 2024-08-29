# run_tests.py

import unittest

# Discover and run all tests in the 'tests' directory
loader = unittest.TestLoader()
suite = loader.discover('tests')

runner = unittest.TextTestRunner()
runner.run(suite)
