#!/usr/bin/env python3

"""A simple search app that works inside Flatpak."""

import gi
gi.require_version('Tracker', '3.0')
from gi.repository import Gio, GLib, GObject, Tracker

import argparse
import logging
import os
import sys

log = logging.getLogger("app")

APPLICATION_ID = "com.example.TrackerSearchApp"


def argument_parser():
    parser = argparse.ArgumentParser(description="Tracker Search App example")
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help="Enable detailed logging to stderr")
    parser.add_argument('query',
                        help="Search query")
    return parser


class TrackerWrapper(GObject.GObject):
    def __init__(self):
        super(TrackerWrapper, self).__init__()
        self._miner_fs_busname = None
        self._miner_fs = None
        self._ready = False

    def start(self):
        self._setup_host_miner_fs()

    @staticmethod
    def _in_flatpak():
        """Indicates if app is running as flatpak"""
        return os.path.exists("/.flatpak-info")

    def _setup_host_miner_fs(self):
        self._miner_fs_busname = "org.freedesktop.Tracker3.Miner.Files"

        log.debug("Connecting to session-wide Tracker indexer at {}".format(self._miner_fs_busname))

        try:
            self._miner_fs = Tracker.SparqlConnection.bus_new(self._miner_fs_busname, None, None)
            log.info("Using session-wide tracker-miner-fs-3")
            self._ready = True
            self.notify('ready')
        except GLib.Error as error:
            log.warning("Could not connect to host Tracker miner-fs at {}: {}".format(self._miner_fs_busname, error))
            if self._in_flatpak():
                self._setup_local_miner_fs()
            else:
                self._ready = None
                self.notify('ready')

    def _setup_local_miner_fs(self):
        self._miner_fs_busname = APPLICATION_ID + ".Tracker3.Miner.Files"
        log.debug("Connecting to bundled Tracker indexer at {}".format(
                  self._miner_fs_busname))

        Gio.bus_get(Gio.BusType.SESSION, None, self._setup_local_bus_connection_cb)

    def _setup_local_bus_connection_cb(self, klass, result):
        bus = Gio.bus_get_finish(result)

        miner_fs_startup_timeout_msec = 30 * 1000
        miner_fs_object_path = "/org/freedesktop/Tracker3/Miner/Files"

        bus.call(
            self._miner_fs_busname, miner_fs_object_path,
            "org.freedesktop.DBus.Peer", "Ping", None, None,
            Gio.DBusCallFlags.NONE, miner_fs_startup_timeout_msec, None,
            self._setup_local_miner_fs_ping_cb)

    def _setup_local_miner_fs_ping_cb(self, klass, result):
        try:
            klass.call_finish(result)
            self._log.info("Using bundled tracker-miner-fs-3")
            self._miner_fs = Tracker.SparqlConnection.bus_new(self._miner_fs_busname, None, None)
            self.ready = True
            self.notify("tracker-available")
        except GLib.Error as error:
            self._log.warning(
                "Could not start local Tracker miner-fs at {}: {}".format(
                    self._miner_fs_busname, error))
            self._miner_fs_busname = None
            self.notify("tracker-available")

    @GObject.Property(type=Tracker.SparqlConnection, flags=GObject.ParamFlags.READABLE)
    def miner_fs(self):
        return self._miner_fs

    @GObject.Property(type=str, flags=GObject.ParamFlags.READABLE)
    def miner_fs_busname(self):
        return self._miner_fs_busname

    @GObject.Property(type=bool, default=False, flags=GObject.ParamFlags.READABLE)
    def ready(self):
        """True if we are ready to talk to Tracker Miner FS"""
        return self._ready


def main():
    args = argument_parser().parse_args()

    if args.debug:
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    def run_search(wrapper, ready):
        cursor = wrapper.miner_fs.query('SELECT nie:url(?r) { ?r a nfo:FileDataObject ; fts:match \"%s\" }' % args.query)
        print("Search results for \"%s\":" % args.query)
        while cursor.next():
            print("  %s" % cursor.get_string(0)[0])

    wrapper = TrackerWrapper()
    wrapper.connect('notify::ready', run_search)
    wrapper.start()


try:
    main()
except RuntimeError as e:
    sys.stderr.write("ERROR: {}\n".format(e))
    sys.exit(1)
