## File Format

All files are saved as `XXXX-apps.txt` where `XXXX` stands for the provider of the IDs for the apps.
All files have timestamp of when they were generated, licensing, default flags and the list of IDs.
File example:
```
# SPDX-License-Identifier: LGPL-2.1-or-later

default-flags=popular,featured
app.drey.Biblioteca
```

Every line starting with `#` is ignored. This also applies to blank lines.

## Default values

Default values can be set at the start of the file (they have to be the first readable lines). Applying default options somewhere
else than at the start of the file, setting more default options than are available or setting 2 or more same default options
in one file will result in  an exception.
Currently there are 2 default options: `default-flags` and `default-categories`. Both work the same way on their
respective columns (explained below).

The option `default-flags=` contains each flag that is applied to IDs without the specified second column. Valid flags
are `popular`, `featured` and `skip`. IDs with `skip` flag are ignored durring the generation of `org.gnome.Apps-list.xml`.
To apply multiple flags use `,` as a delimeter between them. Example: `default-flags=popular,featured`. This default-option
must be used if at least one ID has blank flag column.

The option `default-categories=` contains each category that is applied to IDs without the specified third column. List
of valid categories: https://specifications.freedesktop.org/menu-spec/latest/apa.html, aside from categories available on
this website, there is `Featured` which is also a valid category.
Applying multiple categories is same as above.


## ID Line Format

Lines with IDs use this format:
```
ID  FLAGS   CATEGORIES
```
Each column is seperated by `\t` and can be left blank, in that case default values are used in that column. Lines with
more columns are considered invalid and will result in an exception if the file used in a script. Applying multiple flags and/or
categories is the same as above.
