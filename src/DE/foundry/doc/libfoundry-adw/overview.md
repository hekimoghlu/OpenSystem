Title: Overview

# Overview

FoundryAdw is a [GNOME](https://www.gnome.org/) library which provides
integration between libfoundry and libadwaita.

##  pkg-config name

To build a program that uses Foundry, you can use the following command to get
the cflags and libraries necessary to compile and link.

```sh
gcc hello.c `pkg-config --cflags --libs libfoundry-adw-1` -o hello
```
