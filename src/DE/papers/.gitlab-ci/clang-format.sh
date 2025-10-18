#!/bin/sh

set -e

exitval=0

formattable="$(find libdocument/ libview/ nautilus/ previewer/ thumbnailer/ \( -name "*.c" -o -name "*.h" \))"
unformatted="$(for f in $formattable; do clang-format $f > $f.clang-format && diff -uN $f $f.clang-format || echo "$f"; done)"

if [ ${#unformatted} -ne 0 ]; then
  exitval=1

  echo >&2 "The following files are not correctly formatted:"
  for f in "${unformatted}"; do
    echo "$f" >&2
  done
  echo >&2

  echo "The files above were not correctly formated."
  echo "Please run clang-format or set your editor to respect the"
  echo "configuration in this repository"
fi

exit $exitval
