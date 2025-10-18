#!/bin/sh
# Create a directory of enabled extract rules inside the build tree, for
# use when running tracker-extract from the build tree.

set -ex

if [ "$#" -lt 2 ]; then
    echo >&2 "Usage: $0 SOURCE_DIR TARGET_DIR [RULE1 RULE2...]"
    exit 1;
fi

source_dir=$1
target_dir=$2
shift
shift

if [ ! -d ${target_dir} ]; then
    mkdir -p ${target_dir}
fi

# Start from a clean directory, this is very important when we
# reconfigure an existing build tree.
rm -f ${target_dir}/*.rule

while [ -n "$1" ]; do
    rule="$1"
    cp ${source_dir}/${rule} ${target_dir}
    shift
done
