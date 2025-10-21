import os

AUTHOR = "Yuri Konotopov <ykonotopov@gnome.org>"
SITENAME = "GNOME browser integration"
SITEURL = os.getenv("PELICAN_SITE_URL", "http://localhost:8000")

PATH = "content"

TIMEZONE = "UTC"

DEFAULT_LANG = "en"

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

MARKDOWN = {
    "extension_configs": {
        "markdown.extensions.codehilite": {"css_class": "highlight"},
        "markdown.extensions.extra": {},
        "markdown.extensions.meta": {},
        "markdown.extensions.toc": {
            "title": "Content",
        },
    },
    "output_format": "html5",
}

DEFAULT_PAGINATION = 10

THEME = "themes/gnome"

# Uncomment following line if you want document-relative URLs when developing
# RELATIVE_URLS = True
