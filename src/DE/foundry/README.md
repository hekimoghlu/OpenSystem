# Foundry

This tool aims to extract much of what makes GNOME Builder an IDE into a
library and companion command-line tool.

Why?

Because it seems like there is an opportunity to bring many of the automatic
IDE features of Builder to a command line environment.

To do this, foundry works similar to other developer environments where you
source a bunch of things into your sub-shell. Except, in Foundry's case, there
is a persistent program that lives above that sub-shell which may be interacted
with using the `foundry` commands.

This persistent ancestor process allows for a build manager, LSP management,
SDK tooling, device management and more to run while you are in your shell.

This is born out of the need for me to have much of my Builder tooling
available even when I am not using Builder directly.

Additionally, if/when Builder were to be rebuilt upon Foundry, it can export
the `FOUNDRY_ADDRESS` to a D-Bus socket allowing the use of foundry within
terminal tabs to interact with the Builder instance directly.

A stretch goal is for Ptyxis to be able to have insight into the Foundry
context for new tabs so that multiple tabs can share the same context.


## Documentation

 * [Foundry API Documentation](https://gnome.pages.gitlab.gnome.org/foundry/foundry-1/index.html) can be found here.
 * [FoundryGtk API Documentation](https://gnome.pages.gitlab.gnome.org/foundry/foundry-gtk-1/index.html) can be found here.
 * [FoundryAdw API Documentation](https://gnome.pages.gitlab.gnome.org/foundry/foundry-adw-1/index.html) can be found here.

Foundry heaviy uses Libdex. Knowing how futures work is required. Knowing how fibers work is also useful.

 * [Libdex API Documentation](https://gnome.pages.gitlab.gnome.org/libdex/libdex-1/index.html) can be found here.


## Feature Support

Lots of things are in development, but there is some support for the
following tooling:

### Language Servers

 * astro-ls
 * bash
 * blueprint
 * clangd
 * elixir
 * glsl
 * gopls
 * intelephense
 * jdtls
 * jedi
 * lua
 * mesonlsp
 * pyrefly
 * python-lsp-server
 * ruff
 * rust-analyzer
 * serve-d
 * sourcekit
 * ts/js
 * ty
 * vala
 * vhdl
 * zls

### Build Systems

 * Autotools
 * BuildStream
 * CMake
 * Cargo
 * Dub
 * Go
 * Gradle
 * Make
 * Maven
 * Meson
 * Npm
 * PHPize
 * Waf

### Container Systems

 * Flatpak
 * JHBuild
 * Podman
 * Distrobox
 * Toolbx
 * Host (via sandbox escapes)
 * None (current environment)

### Project Configuration Formats

 * BuildConfig (simple GKeyFile from Builder)
 * Flatpak Manifests (Both JSON and Yaml)

### Device Integration

 * Deviced to communicate to remote devices
 * Qemu-user-static to run non-native architectures

### Documentation

 * Devhelp2 file-format as exposed by gtk-doc and gi-doc

### Version Control Systems

 * Git

### Linters

 * Codespell


## Design

The foundry library is built heavily upon libdex which is a library I wrote
based on more than 2 decades of writing both concurrent and parallel programs
and libraries. Libdex is my third attempt at doing so (after libgtask and
libiris).

It combines futures and fibers in a way that makes async C programming
significantly less annoying. Imagine if Threading Building Blocks, Grand
Central Dispatch, twisted deferreds, GTask, and Microsoft CCR had a baby.

It supports wait-free work-stealing along with work-queues which do not suffer
from thundering herd problems thanks to `io_uring` and `EFD_SEMAPHORE`.

I wish I had written it before writing Builder, and this is an attempt to
redo the Builder internals on such a design.


## Subsystems

You can find various subsystems in the `lib/` directory.

Most of them are `FoundryService` which are managed by the `FoundryContext`
which is your top-level object for a project. Use `FoundryService` futures
for tracking when they are ready before using. That provides the ability
to remove most "state tracking" code in the associated tooling.


## Complexity and Fibers

Many things do not need fibers and creating them would be unnecessary.
However, whenever you have something complex that needs to manage multiple
concurrent tasks you should use them. The result is much easier to read and
the points where you become concurrent are much more obvious.


## Life-cycle Tracking

Most of the API reflects that we'll be using fibers heavily and thus having
your stack ripped out from under you at some point. To make this safer most
of the API implicitly increments the reference count.

API that does not increment reference counts use the `get()` semantics while
those that do use the `dup()` semantics.


## Testing

Since we have this available as a tool with a sub-shell I expect that we will
be able to do much improved testing over what was possible within Builder.
Trying to unit test a build pipeline inside of a UI application is quite a
difficult endeavor.


## Some Tooling Ideas

```
# If you're inside a foundry environment subshell, FOUNDRY_ADDRESS
# will be set which will let foundry command to connect to the
# long running instance and keep persistence between runs.

# Clone a project (default to GNOME/)
foundry clone gnome:gnome-builder
foundry clone freedesktop:mesa/mesa

# Init foundry from existing project
foundry init

# Add a new configuration
foundry config add --flatpak org.gnome.Builder.Devel.json
foundry config remove org.gnome.Builder.Devel.json

# Change active configuration
foundry config switch org.gnome.Builder.Devel.json

# Enter a PTY/Shell with the environment
foundry enter
foundry devenv

# Run default run command
foundry run
foundry run my-command
foundry run -- bash

# Add a run command
foundry commands add my-command -- MY=env http-server -d ./
foundry commands add my-group -c my-command1 -c my-command2

# Debug a command
foundry debug my-command -p named-command-in-group

# Run within valgrind
foundry valgrind my-command

# Build the project
foundry build

# Rebuild the project
foundry rebuild

# Install the project
foundry install

# Clean the project
foundry clean

# Set environ on a command
foundry commands set my-command --env=FOO=BAR

# List available SDKs
foundry sdk list

# Install an SDK
foundry sdk install org.gnome.Sdk//master

# Change SDK of configuration
foundry sdk org.gnome.Sdk//master
foundry sdk host

# Spawn a language server w/ stdin/stdout transport
foundry lsp python
foundry lsp rust

# Index source code
foundry index

# Search the source code for class Object
foundry search -c Object

# Search and replace
foundry replace -e MyObject OtherObject

# Profile a command
foundry profile [--command my-command]

# List available devices
foundry device list

# Set current device
foundry device switch host

# Add a device
foundry device add ssh://user@remote

# Deploy project to target device
foundry deploy

# Export the project (to a Flatpak)
foundry export

# List available unit tests
foundry test list

# Run a test
foundry test run testname

# Alias of foundry test run
foundry test

# Debug a test
foundry test debug testname

# Create a new release
foundry release 48.alpha --branch gnome-48 --bump=post

# Reload foundry state (parent process)
foundry reload

# Rename a gobject class
foundry refactor rename class IdeObject FoundryObject

# Reformat a file
foundry format file.c

# List symbols in a file
foundry symbols file.c

# Create a new project from template
foundry create --template name --language c --license gpl3 --git

# List diagnostics from last build
foundry diagnostics list [--error --warning]

# List pipeline
foundry pipeline info

# Add a command in the pipeline at position 10
foundry pipeline add 10 my-command

# Allow Builder to attach to this instance
foundry listen
```
