#!/usr/bin/bash
FILE_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
wget -nc http://files.grouplens.org/datasets/movielens/ml-100k.zip -P "${FILE_DIR}"
unzip -n "${FILE_DIR}/ml-100k.zip" -d "${FILE_DIR}"