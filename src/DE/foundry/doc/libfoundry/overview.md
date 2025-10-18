Title: Overview

# Overview

Foundry is a [GNOME](https://www.gnome.org/) library and command line tool
to implement IDE functionality across a variety of packging and build systems,
language servers, linters, debuggers, simulators, devices, and more.

## Building

In your C sources, include libfoundry using `<foundry.h>` like the following.

```c
#include <foundry.h>
```

To build a program that uses Foundry, you can use the following command to get
the cflags and libraries necessary to compile and link.

```sh
cc hello.c $(pkg-config --cflags --libs libfoundry-1) -o hello
```

## Version Checks

You can check the version of libfoundry at compile time using the version
checking macros.

```c
#if FOUNDRY_CHECK_VERSION(1, 0, 0)
  /* version specific code here */
#endif
```

## Feature Flags

Foundry has a number of features that are in development and my not be
stablized in the release of Foundry you are consuming. You can check for
feature availability at compile time using the appropriate `#ifdef`.

The feature flags are defined in the `libfoundry-config.h` installed into
your library directory, similar to glib's `glib-config.h`.

```c
#include <foundry.h>

#ifdef FOUNDRY_FEATURE_DEBUGGER
  /* Use Debugger Specific APIs */
#endif
```

In most cases it is advised for applications to control their libfoundry
library and feature flags and that is preferred over conditional checks.
This may mean statically linking libfoundry or building it as a shared
library in a Flatpak.
