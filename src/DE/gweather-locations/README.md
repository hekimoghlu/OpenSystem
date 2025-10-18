GWeather Locations Database
===========================

The GWeather locations database contains a list of locations used by GNOME
components through the [GWeather library][libgweather].

The locations are structured in an XML file, which follows a provided
[schema file](./data/locations.dtd).

The XML source is "compiled" into a binary format for fast parsing and
access.

Location names are translatable.

[libgweather]: https://gitlab.gnome.org/GNOME/libgweather

Using the locations database
----------------------------

The locations database provides a [pkg-config][pkgconfig] file called
`gweather-locations.pc`. The pkg-config file contains the following
variables:

- `locations_xml`: the installation path of the XML file
- `locations_dtd`: the installation path of the DTD schema for the XML file
- `locations_db`: the installation path of the binary file

[pkgconfig]: https://www.freedesktop.org/wiki/Software/pkg-config/

Licensing
---------

The database is released under the terms of the GNU General Public License
version 2.0 or, at your option, any later version.
