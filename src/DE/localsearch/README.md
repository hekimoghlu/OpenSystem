# LocalSearch

LocalSearch is the file search framework of the GNOME desktop. It stores
data about user files structured by the
[Nepomuk definitions](https://gnome.pages.gitlab.gnome.org/tinysparql/ontologies.html#nepomuk),
features a tightly sandboxed metadata extractor, and provides facilities to alter
file metadata.

The data is exposed through a SPARQL endpoint, applications can access
this data via [a portal](https://gnome.pages.gitlab.gnome.org/tinysparql/sandboxing.html).

See the [available documentation](https://gnome.pages.gitlab.gnome.org/localsearch).

# Contact

LocalSearch is a free software project developed in the open by contributors. You
can make questions about it at:

  * [Matrix](https://matrix.to/#/#tracker:gnome.org)
  * [GNOME Discourse](https://discourse.gnome.org/tag/tracker)

# Reporting issues

If you found an issue in LocalSearch, a bug report at
https://gitlab.gnome.org/GNOME/localsearch/issues will be welcome. Please
see the [GNOME bug reporting guidelines](https://handbook.gnome.org/issues/reporting.html)
for reported bugs.

# Contributing

Your contributions are greatly appreciated! As LocalSearch uses the Meson
build system, the [GNOME handbook](https://handbook.gnome.org/development/building.html)
greatly applies to it. LocalSearch will typically be a D-Bus service running in the
host, you can consider the Toolbx/JHBuild approaches to build it individually.

For more information on the code itself, see the [hacking documentation](HACKING.md).
