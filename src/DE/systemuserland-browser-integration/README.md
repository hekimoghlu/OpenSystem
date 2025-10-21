# GNOME browser integration website
## Introduction

This repository contains static site generator for GNOME browser integration.  
Website content was migrated from the [GNOME wiki](https://wiki.gnome.org/Projects/GnomeShellIntegration) which will be retired.

## Build and development

The [Pelican](https://getpelican.com/) static site generator is used in this repository.

This repository contains [devcontainer](https://containers.dev/) configuration which will create ready to go development workspace in one click if you use VSCode or IntelliJ IDE. In VSCode you will get "Run" button to live-preview any changes.

To build website manually you will need Python 3 (look to the .devcontainer/docker-compose.yml for exact version).

To install build dependencies run:

```
pip install -r requirements.txt
```

To build website:

```
python -m pelican
```
