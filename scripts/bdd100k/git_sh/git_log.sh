#!/bin/bash
git_root=$(git rev-parse --show-toplevel | head -1)

git_out="$1"
if [ -z "$git_out" ]; then
    git_out="$git_root/git_out"
    mkdir -p "$git_out"
fi

pushd "$git_root"
git status > "$git_out/git_status.txt"
git log > "$git_out/git_log.txt"
git diff HEAD > "$git_out/git_diff.txt"
git rev-parse HEAD > "$git_out/git_head.txt"
popd
