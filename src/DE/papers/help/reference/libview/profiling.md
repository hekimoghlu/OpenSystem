Title: Profiling

Papers makes use of
[Sysprof](https://developer.gnome.org/documentation/tools/sysprof.html) for
profiling. To do profiling on Papers, it is first necessary to build papers in
debug mode: `meson setup -Ddebug=true`, and to have the latest developer headers
for `sysprof` installed.

## Profiling in action

Once compiled having Sysprof available, Papers can be started through Sysprof.
The most relevant information profiled to date is related to `PpsJobs`, that
will be available under `Timings`.
