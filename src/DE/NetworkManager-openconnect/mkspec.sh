#!/bin/bash

TARGET="$1"

GITCOMMIT="$(git rev-parse HEAD)"
GITDESC="$(git describe --tags HEAD)"

GITTAG="$(echo $GITDESC | sed 's/-[0-9]\+-g[0-9a-f]\+$//')"
GITTAG="${GITTAG#v}"
GITTAG="${GITTAG%-dev}"

if [ "v$GITTAG" = "$GITDESC" ]; then
    ISSNAP=0
    GITCOUNT=0
else
    ISSNAP=1
    GITCOUNT="$(echo $GITDESC | sed 's/.*-\([0-9]\+\)-g[0-9a-f]\+$/\1/')"
fi

sed -e "s/@ISSNAP@/${ISSNAP}/" \
    -e "s/@VERSION@/${GITTAG}/" \
    -e "s/@SNAPCOMMIT@/${GITCOMMIT}/" \
    -e "s/@SNAPCOUNT@/${GITCOUNT}/" \
    ${TARGET}.spec.in > ${TARGET}.spec
