# Overview

LocalSearch offers a series of D-Bus services to index filesystem
resources, and expose this data so it can be queried in arbitrary
manners from consumers. LocalSearch is the indexer and search engine
that powers desktop search for core GNOME components.

It consists of the following daemons:

- `localsearch-3`: Main indexer process and [SPARQL endpoint](endpoint.md)
  for RDF data. This process monitors the filesystem and ensures
  that data is synchronized with the filesystem structure.

- `localsearch-extractor-3`: Helper process to extract metadata from file content.
  This process gets restricted capabilities through [seccomp](https://en.wikipedia.org/wiki/Seccomp).

- `localsearch-control-3`: Allows limited control on the indexed locations
  and data. This process is separated from `localsearch-3` so it can be used
  from within [Flatpak](https://flatpak.org) sandboxes, see its
  [D-Bus API](dbus-api.md#orgfreedesktoplocalsearch3control) for more information.

- `localsearch-writeback-3`: Utility daemon that updates file metadata out of its
  RDF description. This service only starts on requests from applications, see its
  [D-Bus API](dbus-api.md#orgfreedesktoplocalsearch3writeback) for more information.

LocalSearch is friendly to [sandboxing](https://gnome.pages.gitlab.gnome.org/tracker/docs/developer/sandboxing.html).
The data is segmented in a way that different types of consumers may only
get the portions of data that are relevant to them. See how [this works](endpoint.md#graphs).

LocalSearch also comes with a collection of [command line utilities](commandline.md).
