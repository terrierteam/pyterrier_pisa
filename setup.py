import shutil
from pathlib import Path
import sys
import os
from setuptools import find_packages
from skbuild import setup
import skbuild
import zipfile
import numpy as np


class bdist_wheel(skbuild.command.bdist_wheel.bdist_wheel):
  def run(self):
    res = super().run()
    wheel = self.distribution.dist_files[0][2]
    pisathon_so = list((Path(self.distribution.package_dir['pyterrier_pisa']).parent.parent).glob('**/_pisathon*.so'))[0]
    lib_tbb = list((Path(self.distribution.package_dir['pyterrier_pisa']).parent.parent.parent).glob('**/libtbb.so.2'))[0]
    print(f'patching wheel with {pisathon_so} and {lib_tbb}')
    base_path = Path('/tmp/libtbb' if os.environ.get("PT_PISA_MANYLINUX", "False") == "True" else '_skbuild/libtbb')
    base_path.mkdir(exist_ok=True, parents=True)
    shutil.copy(lib_tbb, base_path/lib_tbb.name)
    shutil.copy(pisathon_so, base_path/pisathon_so.name)
    return res

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pyterrier_pisa",
    version="0.0.4" + os.environ.get('PT_PISA_VERSION_SUFFIX', ''),
    description="A PyTerrier interface to the PISA search engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Sean MacAvaney',
    license="",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=['python-terrier>=0.8.0', 'numpy>=1.21.0'],
    python_requires=">=3.7",
    entry_points={
        'console_scripts': ['pyterrier_pisa=pyterrier_pisa:main'],
    },
    cmdclass={'bdist_wheel': bdist_wheel}
)
