# Gtkmm's website

This is the source of the [gtkmm website](https://gtkmm.gnome.org).

The source consists of a DocBook 5.0 document.
It requires the docbook5-xml and docbook-xsl-ns packages (Ubuntu names,
can have other names in other distros). It may be possible to build without
these packages, but it will be __much__ slower (minutes instead of seconds).[^1]

The website can be built with Meson or Autotools.

When a commit is pushed to the master branch in the git repository, the website
is built and published at https://gnome.pages.gitlab.gnome.org/gnomemm-website
and https://gtkmm.gnome.org.

# General information

Web site
 - https://gtkmm.gnome.org
 - https://gnome.pages.gitlab.gnome.org/gnomemm-website

Discussion on GNOME's discourse forum
 - https://discourse.gnome.org/tag/cplusplus
 - https://discourse.gnome.org/c/platform

Git repository
 - https://gitlab.gnome.org/GNOME/gnomemm-website

Bugs can be reported to
 - https://gitlab.gnome.org/GNOME/gnomemm-website/-/issues

Patches can be submitted to
 - https://gitlab.gnome.org/GNOME/gnomemm-website/-/merge_requests

# Building

## Building with Meson

- Create a build directory and configure it:
  ```sh
  cd gnomemm-website
  meson setup [options] <build-dir>
  ```
  Do not call the build-dir gnomemm-website/build. There is already such a
  directory, used when building with Autotools.

- Create the html files of the website:
  ```sh
  cd <build-dir>
  ninja
  ```

- Watch the result:

  Open \<build-dir>/docs/html/index.html in a web browser.

- Publish the website at gtkmm.gnome.org:

  Done automatically when new git commits are pushed to the master branch.

  There is a `publish` target (requiring a login at www.gtkmm.org),
  but it is disabled and there is no reason to use it.

## Building with Autotools

- Configure, build in the source directory:
  ```sh
  cd gnomemm-website
  ./autogen.sh
  ```

- Create the html files of the website:
  ```sh
  make
  ```

- Watch the result:

  Open docs/html/index.html in a web browser.

- Publish the website at gtkmm.gnome.org:

  Done automatically when new git commits are pushed to the master branch.

  There is a `post-html` target (requiring a login at www.gtkmm.org),
  but there is no reason to use it.

## Adding a new language

- Generate a temporary .pot file and the .po file for your locale, say xx_XX:
  ```sh
  cd docs
  itstool -o gnomemm-website.pot C/index.docbook C/*.xml
  mkdir xx_XX
  msginit -i gnomemm-website.pot -o xx_XX/xx_XX.po -l xx_XX.utf8
  ```

- Translate the .po file.

- Modify docs/C/index.docbook and docs/C/index.withimg:

  Find `<formalpara xml:id="language-menu">`, and add your locale.
  ```xml
  <listitem>
    <para><link its:translate="no" xlink:href="../xx_XX/index.html">Xxxx</link></para>
  </listitem>
  ```

- Add your language to docs/LINGUAS (used when building with Meson).

- Add your language to HELP_LINGUAS in docs/Makefile.am (used when building with Autotools).

- Run `make` or `ninja` to generate all pages.

--------------------
<!-- footnote -->
[^1]: The `xmllint` command is told to read files from `http://docbook.org`.
The `xsltproc` command is told to read files from `http://docbook.sourceforge.net`.
The commands first search for local copies of those files. If local copies exist
and are installed at expected locations, the commands make no network accesses.
