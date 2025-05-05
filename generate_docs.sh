#!/bin/bash -e

##########################################
# Generate API documentation.
# Usage:
#   generate_docs.sh [--use-current-env]
##########################################

ROOT=$(dirname "${BASH_SOURCE[0]}")
DOCS_DIR=$ROOT/docs

if [ "$1" ];then
  if [ "$1" != "--use-current-env" ];then
    echo Unknown flag passed $1. Usage: generate_docs.sh [--use-current-env]
    exit 1
  fi
else
  ENV_DIR=$(mktemp -d)
fi

if [ "$ENV_DIR" ];then
  echo Building temp env $ENV_DIR
  python3 -m venv $ENV_DIR
  source $ENV_DIR/bin/activate
  pip install -e $ROOT[tf,torch] pdoc --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
else
  echo Using current env
fi

echo Generating docs
cd $ROOT
pdoc --docformat google -o $DOCS_DIR ./edgemdt_cl/{keras,pytorch} --no-include-undocumented --no-search -t ./pdoc/template

if [ "$ENV_DIR" ];then
  echo Removing $ENV_DIR
  rm -rf $ENV_DIR
fi
