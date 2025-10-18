# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = "GNOME Python API"
author = "GNOME Developers"
copyright = "2023"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
]

# include_patterns = ["build/source/**"]
templates_path: list[str] = ["./sphinx"]
exclude_patterns: list[str] = []

# Default role for backtick text `like this`.
default_role = "py:obj"
add_module_names = False
# toc_object_entries = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
pygments_style = "tango"
html_theme = "pydata_sphinx_theme"
html_show_copyright = False
html_title = project

# Ensure mirrors and forks point to one canonical source
html_baseurl = "https://api.pygobject.gnome.org/"

html_theme_options = {
    "globaltoc_maxdepth": 2,
    "navigation_depth": 3,
    "footer_center": ["genindex"],
    "show_prev_next": False,
    "content_footer_items": ["copyright", "last-updated"],
    "header_links_before_dropdown": 6,
}

html_static_path = ["static"]

html_css_files = ["custom.css"]

# -- Intersphinx

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "pycairo": ("https://pycairo.readthedocs.io/en/latest/", None),
}
