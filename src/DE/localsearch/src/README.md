The project tree is structured the following way:

- `cli` contains the command line utilities installed as `localsearch`
- `common` contains the private static library common between the rest of
  components.
- `control` contains the indexer control service. See the
  [project documentation](https://gnome.pages.gitlab.gnome.org/localsearch/dbus-api.html#orgfreedesktoplocalsearch3control)
- `extractor` contains the modular file metadata extractor
- `indexer` contains the filesystem structure indexer and SPARQL D-Bus endpoint.
  See the [project documentation](https://gnome.pages.gitlab.gnome.org/localsearch/endpoint.html)
- `writeback` contains the metadata writeback service. See the
  [project documentation](https://gnome.pages.gitlab.gnome.org/localsearch/dbus-api.html#orgfreedesktoplocalsearch3writeback)
