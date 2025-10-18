Title: Overview

# Overview

Spelling is a [GNOME](https://www.gnome.org/) library that provides spellchecking
for `GtkTextView` widgets.

##  pkg-config name

To build a program that uses Spelling, you can use the following command to get
the cflags and libraries necessary to compile and link.

```sh
gcc hello.c `pkg-config --cflags --libs libspelling-1` -o hello
```
