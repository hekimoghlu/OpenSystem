# Foundry from Flatpak

You'll want to include the "libfoundry.json" from your "modules" section of the Flatpak.

You'll also probably want this in your manifest:

```json
    "finish-args" : [
        "--allow=devel",
        "--device=dri",
        "--filesystem=host:ro",
        "--filesystem=~/.local/share/flatpak",
        "--filesystem=/var/lib/flatpak",
        "--filesystem=/var/tmp",
        "--share=ipc",
        "--share=network",
        "--socket=fallback-x11",
        "--socket=wayland",
        "--system-talk-name=org.freedesktop.PolicyKit1",
        "--system-talk-name=org.freedesktop.Flatpak.SystemHelper",
        "--talk-name=org.freedesktop.Flatpak"
    ],
```

Additionally, this is needed to allow ostree to build.

```
    "build-options" : {
        "env" : {
            "V" : "1",
            "BASH_COMPLETIONSDIR" : "/app/share/bash-completion/completions",
            "MOUNT_FUSE_PATH" : "../tmp/"
        }
    },
```
