# Contributing to Manuals

Start by installing [GNOME Builder](https://flathub.org/apps/org.gnome.Builder).

Then clone [this repository](https://gitlab.gnome.org/chergert/manuals.git) from the clone dialog.

When the application opens, click the Run button at the top to build/run a local build of Manuals.

## Dependencies

This application is targeting Flatpak as the distribution method.
Therefore, it depends on being able to install developer SDK documentation via Flatpak.

Builder will pull in all the dependencies for you.

### Gom

You can parse `.devhelp2` XML files quite fast with no database and in the past that is what we've done.
However, when you start working with multiple SDK versions it quickly becomes a performance problem.
In Manuals, the Gom project is used to (de)serialize GObjects to and from a SQLite database on demand.

### Libpanel

The styling with resizeable sidebars comes from libpanel (and ultimately GNOME Builder).

### WebKit

Browsing capabilities are provided by WebKit.
We do some alterations of data in the webview to make theming integrate better.

In Builder, styling comes from the underlying `GtkSourceStyleScheme` but this is not done in Manuals.

## Upstreaming Things

Because of GNOME Release Team not wanting applications to depend on applications, some code is duplicated with Builder.
We need to occasionally keep that code in sync to avoid duplicating bugs.

There is the possibility of making Manauls export a shared library, but where we want the lines there set for integration is up in the air.

## Coding Style

Please follow the GTK coding style.

When in doubt, look at the surrounding code.
