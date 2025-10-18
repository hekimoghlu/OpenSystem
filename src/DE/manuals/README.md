# Manuals

Install, Browse, and Search developer documentation

Manuals is an extraction of the Documentation component of GNOME Builder
into a standalone application.

## Feature Requests and Design Changes

Manuals uses the issue tracker for tracking engineering defects only.

Feature requests tend to be long, drawn out, and never fully solvable.
Therefore we request that you file an issue with the
[Whiteboards project](https://gitlab.gnome.org/Teams/Design/whiteboards/)
and progress towards designing the feature you'd like to see. I will not design
the feature for you.

The outcome of the design process should be a specification which includes:

 * How the feature should work
 * How the feature should not work
 * How the feature interacts with the existing features
 * If any existing features should be changed or removed
 * Any necessary migration strategies for existing users
 * UI mock-ups (if necessary)
 * How the feature should be tested
 * What are the risks and security gotchas involved?
 * Who is going to implement the feature

After that process has completed, you may file an issue referencing it.

## Installation

Currently you need to install by grabbing a Flatpak artifact from CI.

You can install documentation manually using the `org.gnome.Sdk.Docs`
runtimes (or similar) or use the application to install them.

```sh
flatpak install --user gnome-nightly org.gnome.Sdk.Docs//master
flatpak install --user flathub org.gnome.Sdk.Docs//45
```

## Dependencies

 * GLib/GObject/Gio/etc
 * libfoundry

## How it works

Manuals will scan your host operating system, flatpak runtimes, and jhbuild
installation for documentation. Currently, the devhelp2 format is supported
but additional formats may be added in the future.

The documentation is indexed in SQLite using GNOME/gom.

If the etag for any documentation has changed during startup, Manuals will
purge the existing indexed contents and re-index that specific documentation.

## Future Work

 * I'd love to see idexing of manpages such as POSIX headers.
 * Sphinx documentation format used by Builder and GNOME HIG
 * Indexing online-based documentation

## Code of Conduct

When interacting with the project, the [GNOME Code Of Conduct](https://conduct.gnome.org/) applies.

## Screenshots

![Empty](https://gitlab.gnome.org/GNOME/manuals/-/raw/main/data/screenshots/empty.png)

![Browse](https://gitlab.gnome.org/GNOME/manuals/-/raw/main/data/screenshots/browse.png)

![Search](https://gitlab.gnome.org/GNOME/manuals/-/raw/main/data/screenshots/search.png)

![Installing SDKs](https://gitlab.gnome.org/GNOME/manuals/-/raw/main/data/screenshots/install.png)

![Dark](https://gitlab.gnome.org/GNOME/manuals/-/raw/main/data/screenshots/dark.png)

![Mobile Friendly Search](https://gitlab.gnome.org/GNOME/manuals/-/raw/main/data/screenshots/mobile-search.png){width=482px}

![Mobile Friendly Reading](https://gitlab.gnome.org/GNOME/manuals/-/raw/main/data/screenshots/mobile-display.png){width=482px}
