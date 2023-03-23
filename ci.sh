#!/bin/bash

set -eux -o pipefail

EPIC_LIB='sklearn'

# Log some general info about the environment
uname -a
env | sort

# print some general information about our python environment
python -c "import sys, struct, ssl; print('#' * 70); print('python:', sys.version); print('version_info:', sys.version_info); print('bits:', struct.calcsize('P') * 8); print('openssl:', ssl.OPENSSL_VERSION, ssl.OPENSSL_VERSION_INFO); print('#' * 70)"

#
# build and install the library
#

# this is needed to make sure "git describe" is completely accurate under github action workflow
# see https://github.com/actions/checkout/issues/290#issuecomment-680260080
git fetch --tags --force
python -m pip install -U 'quicklib>=2.4'
quicklib-setup sdist --formats=zip
python -m pip install dist/*.zip --extra-index-url https://d2dsindf03djlb.cloudfront.net

#
# run the tests
#

rm -rf workdir && mkdir workdir && cd workdir
pip install -r ../test_requirements.txt
EPIC_INSTALL_DIR=$(python -c "import epic; print(epic.__path__[0])")
ln -s "${EPIC_INSTALL_DIR}" epic
pytest -r a --verbose --import-mode=importlib "epic/${EPIC_LIB}" \
  --cov=epic.${EPIC_LIB} --cov-report html:cov_html --cov-report term \
  ${PYTEST_ARGS:-}
if (which zip); then
  zip -r cov_html.zip cov_html
fi
