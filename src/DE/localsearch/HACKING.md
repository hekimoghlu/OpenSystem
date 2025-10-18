# LocalSearch HACKING guide

This file contains information useful for developers who want to improve
LocalSearch.

## Automated testing

You can run the LocalSearch testsuite using the `meson test` command.

LocalSearch main GitLab repository runs the tests automatically in CI. The
`.gitlab-ci.yml` file controls this. Here are the latest tests that were run:

 * https://gitlab.gnome.org/GNOME/localsearch/pipelines

Most distros also run the test suite as part of their building process.

## Logging

The following environment variables control logging from LocalSearch daemons:

  * `G_MESSAGES_DEBUG`: controls log output from all GLib-based libraries
    in the current process. Use `G_MESSAGES_DEBUG=all` to see every logging
    message.
  * `LOCALSEARCH_DEBUG`: takes a comma-separated list of keywords to enable
    extra debugging output. Use the keyword 'help' for a list of keywords.

    The options include:
      - `config`
      - `decorator`
      - `miner-fs-events`
      - `monitors`
      - `statistics`
      - `status`
      - `sandbox`

    See the relevant `man` page for options relevant to the localsearch-3 daemon.

You can set these variables when using `tracker-sandbox`, and when running the
Tracker test suite. Note that Meson will not print log output from tests by
default, use `meson test --verbose` or `meson test --print-errorlogs` to
enable.

You can use `LOCALSEARCH_DEBUG=tests` to see logging from the test harness,
including full log output from the internal D-Bus daemon for functional-tests.
Note that by default, functional tests filter output from the D-Bus daemon to
only show log messages from Tracker processes. Anything written directly to
stdout, for example by `g_print()` or by the dbus-daemon itself, will not be
displayed unless `LOCALSEARCH_DEBUG=tests` is set.

When working with GitLab CI, you can use the
[Run Pipeline dialog](https://gitlab.gnome.org/GNOME/tracker/pipelines/new)
to set the values of these variables and increase the verbosity of the tests in
CI.

## Instrumenting LocalSearch daemons

LocalSearch daemons are usually not started directly, instead they are started
by Systemd or D-Bus. In order to run the indexer fully under e.g. gdb or valgrind,
you will need stop it first, e.g.:

```sh
$ localsearch3 daemon --terminate
$ valgrind --leak-check=full --trace-children /usr/libexec/localsearch-3
```

Note that the metadata extractor process is fully managed by the indexer, you
will need to trace both as in the example above.
