import zipfile
import argparse
import re
import tempfile
import os
import sys
from pathlib import Path
import subprocess

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('source')
  args = parser.parse_args()
  source = list(Path(args.source).glob('pyterrier_pisa*.whl'))[0]
  version = re.search(r'cp([0-9]+)', str(source)).group(1)
  print('------------- version ----------------')
  print(version)
  # with tempfile.NamedTemporaryFile() as tmpf:
    # with zipfile.ZipFile(source, 'r') as zipf:
      # for file in zipf.infolist():
        # if re.match(r'pyterrier_pisa.libs/libtbb.*.so.2', file.filename):
          # n_tbb = file.filename

  if os.environ.get("PT_PISA_MANYLINUX", "False") == "True":
    # tbb = Path('/tbb/tbb/lib/intel64/gcc4.8/libtbb.so.2')
    pisathon_so = list(Path('/tmp/libtbb/').glob('_pisathon*-' + version + '-*'))[0]
  else:
    # tbb = list(Path('_skbuild/libtbb/').glob('libtbb.so.2'))[0]
    pisathon_so = list(Path('_skbuild/libtbb/').glob('_pisathon*-' + version + '-*'))[0]
  subprocess.run(['patchelf', '--set-rpath', '$ORIGIN/../../..', str(pisathon_so)])
  with zipfile.ZipFile(source, 'a', zipfile.ZIP_DEFLATED) as zipf:
    n = pisathon_so.name
    zipf.write(pisathon_so, f"pyterrier_pisa/{n}")
    # zipf.write(tbb, 'pyterrier_pisa.libs/libtbb.so.2')

if __name__ == '__main__':
  main()
