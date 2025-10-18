# Miscellaneous D-Bus services

Besides the SPARQL endpoint for data, the filesystem indexer
offers other miscellaneous services for control and additional
features. Access to these services may be allowed from Flatpak
sandboxes.

## org.freedesktop.LocalSearch3.Control

This service allows a certain amount of control on the
indexed locations. This service offers the following two
interfaces:

### org.freedesktop.Tracker3.Miner.Files.Index

Interface to change the indexed locations. It has the following
method:

```
IndexLocation (IN  s  file_uri,
               IN  as graphs,
               IN  as flags)
```

This method requests to index a certain location. The filesystem
indexer will ensure the location is indexed if it was not previously.
the request will persist as long as the caller D-Bus name remains
present in the bus.

The `file_uri` argument expresses the location to be indexed, the
`graphs` array expresses the [data graphs](endpoint.md#graphs) of
interest for the caller (e.g. `tracker:Audio`). The filesystem
indexer may prioritize these files to be available first.

The `flags` argument is unused at the moment and should remain empty.

See the definition of [this interface](https://gitlab.gnome.org/GNOME/localsearch/blob/main/src/control/org.freedesktop.Tracker3.Miner.Files.Index.xml).

### org.freedesktop.Tracker3.Miner.Files.Proxy

Interface to introspect all indexed locations. It has the following
properties:

```
Graphs (as): read only
IndexedLocations (as): read only
```

The global set of indexed locations and graphs will be returned by
these properties.

See the definition of [this interface](https://gitlab.gnome.org/GNOME/localsearch/blob/main/src/control/org.freedesktop.Tracker3.Miner.Files.Proxy.xml).

## org.freedesktop.LocalSearch3.Writeback

This service allows performing changes to the metadata
contained in files out the metadata descriptions in RDF. This service
exposes a single D-Bus interface, with the following method:

```
Writeback (IN  a{sv}  rdf)
```

The `rdf` argument expresses RDF data corresponding to the file
whose metadata is being modified, in the same format that it would be
accepted by the SPARQL endpoint. This can be obtained through e.g.
[tracker_resource_serialize()](https://gnome.pages.gitlab.gnome.org/tracker/docs/developer/method.Resource.serialize.html).

This method may raise an error if the metadata could not be written.

Writeback is only available for audio formats handled by GStreamer,
and XMP metadata.

See the definition of [this interface](https://gitlab.gnome.org/GNOME/localsearch/blob/main/src/writeback/tracker-writeback.xml).
