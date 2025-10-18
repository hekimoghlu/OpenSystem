# Example search app

This is a simple example of building an app that searches user data using Tracker.

## Running on the host

You can run it on your host as follows:

    $ python3 ./example-app.py TEST

This will return all files indexed by Tracker Miner FS which match the word "TEST".

## Building a Flatpak

The provided Flatpak manifest will produce a self-contained Flatpak app, which
works on systems even where Tracker Miner FS isn't installed.

You can build and install the app as follows:

    $ flatpak-builder --install --user ./build ./org.example.TrackerSearchApp.json

You may need to install the nightly GNOME SDK first, or you can modify the
manifest to use a stable version of the GNOME SDK.

You can then run the app:

    $ flatpak run org.example.TrackerSearchApp TEST
