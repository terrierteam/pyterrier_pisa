#!/bin/bash
python setup.py bdist_wheel
python patcher.py dist/ 
pip uninstall -qy pyterrier_pisa
pip install dist/*.whl
