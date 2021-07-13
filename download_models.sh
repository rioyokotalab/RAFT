#!/bin/bash
git_root=$(git rev-parse --show-toplevel | head -1)
pushd "$git_root"

wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip

popd
