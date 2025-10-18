#!/bin/sh
# Post-install script for install stuff that Meson doesn't support directly.
#
# We can't pass the necessary variables directly to the script, so we
# substitute them using configure_file(). It's a bit of a Heath Robinson hack.

set -e

dbus_services_dir="$1"
tracker_miner_services_dir="$2"
domain_prefix="$3"
have_tracker_miner_fs="$4"

mkdir -p ${DESTDIR}/${tracker_miner_services_dir}
if ([ "$have_tracker_miner_fs" = "true" ]); then
  ln -sf "${dbus_services_dir}/${domain_prefix}.Tracker3.Miner.Files.service" "${DESTDIR}/${tracker_miner_services_dir}/"
fi
