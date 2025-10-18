# GNOME app list

This project provides app recommendation data for the GNOME project, in the
form of AppStream data which is installed in the standard system location. This
is mainly used by the [Software](https://gitlab.gnome.org/GNOME/gnome-software)
app.

The goals of the project are to:

1. Ensure that high-quality apps are recommended.
1. Ensure that the recommended apps are kept up to date.
1. Allow app recommendations to be maintained and released separately from
   gnome-software.

There is more context available in the discussion on
[this issue](https://gitlab.gnome.org/GNOME/gnome-software/-/issues/1982#note_1633421).

## Project overview

The most interesting parts of gnome-app-list are:

* `/data`: contains lists of apps:
  * `flathub-apps.txt`: recommended apps pulled from 
     [Flathub](https://flathub.org)
  * `gnome-apps.txt`: [GNOME Circle](https://gitlab.gnome.org/Teams/Circle/)
     apps
  * `other-apps.txt`: a manually curated set of apps (currently empty) 
* `/scripts`:
  * `update_apps.py`: updates the app lists based on data from Flathub and GNOME
    Circle
  * `xml_generator.py`: generates AppStream data from the app lists

## App tagging

Each of the files in the ``/data`` directory contains a list of apps. Each file
also includes configuration for tags that are applied to each app. These tags
are then translated to the outputted AppStream XML which is read by
gnome-software, and determines where in the UI each app is shown.

There are two types of tags: flags and categories.

| Flag / Category    | Outputted XML tag                                                       | UI destination                                                | Minimum No. Required* |
|--------------------|-------------------------------------------------------------------------|---------------------------------------------------------------|-----------------------|
| flag: popular      | `<kudos><kudo>GnomeSoftware::popular</kudo></kudos>`                    | Banners on category pages, editor's picks on the Explore page | 1, 6 |
| flag: featured     | `<custom><value key="GnomeSoftware::FeatureTile">True</value></custom>` | Banners on explore pages                                      | 5 |
| category: featured | `<categories><category>Featured</category></categories>`                | Editor's picks on category pages                              | 3 |

\* This is the number of apps that must be available to make the relevant UI section be visible in the Software app.

See [vendor-customisation.md](https://gitlab.gnome.org/GNOME/gnome-software/-/blob/main/doc/vendor-customisation.md?ref_type=heads) for more details about metainfo tags.

## How to update gnome-app-list

To update `data/flathub-apps.txt` run in the project root:
```
python3 ./scripts/update_apps.py flathub
```

To update `data/gnome-apps.txt` run in the project root:
```
python3 ./scripts/update_apps.py gnome
```

## How to test

Rough instructions for how to test how Software performs with a particular
set of AppStream data files:

* Generate the AppStream XML locally from the app lists:
  `python3 ./scripts/xml_generator.py data/flathub-apps.txt data/gnome-apps.txt
  data/other-apps.txt`. This will generate `org.gnome.App-list.xml`.
* Create a VM with a recent version of GNOME. Then, in the VM:
  *  Clear out the old app recommendations data:
      * cd `/usr/share/swcatalog/xml/` and remove the XML files
      * `rm -rf ~/.cache/gnome-software/`
  * Copy the newly generated XML from gnome-app-list to
    `/usr/share/swcatalog/xml/`
  * Restart Software: `gnome-software --quit && gnome-software`
