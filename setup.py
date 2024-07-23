import shutil
from pathlib import Path
import os
from setuptools import find_packages
from skbuild import setup
import skbuild


def get_version(path):
    for line in open(path, 'rt'):
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


class bdist_wheel(skbuild.command.bdist_wheel.bdist_wheel):
  def run(self):
    res = super().run()
    # wheel = self.distribution.dist_files[0][2]
    pisathon_so = list((Path(self.distribution.package_dir['pyterrier_pisa']).parent.parent).glob('**/_pisathon*.so'))[0]
    # lib_tbb = list((Path(self.distribution.package_dir['pyterrier_pisa']).parent.parent.parent).glob('**/libtbb.so.2'))[0]
    # print(f'patching wheel with {pisathon_so} and {lib_tbb}')
    print(f'patching wheel with {pisathon_so}')
    base_path = Path('/tmp/libtbb' if os.environ.get("PT_PISA_MANYLINUX", "False") == "True" else '_skbuild/libtbb')
    base_path.mkdir(exist_ok=True, parents=True)
    # shutil.copy(lib_tbb, base_path/lib_tbb.name)
    shutil.copy(pisathon_so, base_path/pisathon_so.name)
    return res

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pyterrier_pisa",
    version=get_version("src/pyterrier_pisa/__init__.py") + os.environ.get('PT_PISA_VERSION_SUFFIX', ''),
    description="A PyTerrier interface to the PISA search engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Sean MacAvaney',
    license="",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=list(open('requirements.txt')),
    python_requires=">=3.8",
    entry_points={
        'console_scripts': ['pyterrier_pisa=pyterrier_pisa.cli:main'],
        'pyterrier.artifact': [
            'sparse_index.pisa = pyterrier_pisa:PisaIndex',
        ],
        'pyterrier.artifact.metadata_adapter': [
            'sparse_index.pisa = pyterrier_pisa.pisa_metadata_adapter:pisa_artifact_metadata_adapter',
        ],
    },
    cmdclass={'bdist_wheel': bdist_wheel}
)
