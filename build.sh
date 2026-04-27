#!/bin/bash
python -m build --wheel
pip uninstall -qy pyterrier_pisa
pip install dist/*.whl
