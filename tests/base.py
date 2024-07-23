import unittest
import os
import pyterrier as pt

import tempfile
import shutil
import os

class BaseTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(BaseTestCase, self).__init__(*args, **kwargs)
        if not pt.started():
            pt.init()
        self.here = os.path.dirname(os.path.realpath(__file__))

    def skip_windows(self):
        if BaseTestCase.is_windows():
            self.skipTest("Test disabled on Windows")

    @staticmethod
    def is_windows() -> bool:
        import platform
        return platform.system() == 'Windows'

class TempDirTestCase(BaseTestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        try:
            shutil.rmtree(self.test_dir)
        except:
            pass
