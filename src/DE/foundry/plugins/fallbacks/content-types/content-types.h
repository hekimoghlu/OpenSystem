/* content-types.h
 *
 * Copyright 2025 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include <foundry.h>

G_BEGIN_DECLS

static const struct {
  const char *language;
  const char * const *globs;
  const char * const *content_types;
} languages[] = {
  {
    "r",
    FOUNDRY_STRV_INIT ("*.R", "*.Rhistory", "*.Rout", "*.Rout.fail", "*.Rout.save", "*.Rt", "*.r"),
    FOUNDRY_STRV_INIT ("text/x-R")
  },
  {
    "abnf",
    FOUNDRY_STRV_INIT ("*.abnf"),
    NULL
  },
  {
    "actionscript",
    FOUNDRY_STRV_INIT ("*.as"),
    FOUNDRY_STRV_INIT ("text/x-actionscript")
  },
  {
    "ada",
    FOUNDRY_STRV_INIT ("*.adb", "*.ads"),
    FOUNDRY_STRV_INIT ("text/x-ada", "text/x-adasrc")
  },
  {
    "ansforth94",
    FOUNDRY_STRV_INIT ("*.4th", "*.forth"),
    FOUNDRY_STRV_INIT ("text/x-forth")
  },
  {
    "asciidoc",
    FOUNDRY_STRV_INIT ("*.adoc", "*.asciidoc"),
    FOUNDRY_STRV_INIT ("text/asciidoc")
  },
  {
    "asp",
    FOUNDRY_STRV_INIT ("*.asp"),
    FOUNDRY_STRV_INIT ("application/x-asap", "application/x-asp", "text/x-asp")
  },
  {
    "automake",
    FOUNDRY_STRV_INIT ("GNUmakefile.am", "Makefile.am"),
    NULL
  },
  {
    "awk",
    FOUNDRY_STRV_INIT ("*.awk"),
    FOUNDRY_STRV_INIT ("application/x-awk")
  },
  {
    "bennugd",
    FOUNDRY_STRV_INIT ("*.prg"),
    NULL
  },
  {
    "bibtex",
    FOUNDRY_STRV_INIT ("*.bib"),
    FOUNDRY_STRV_INIT ("text/x-bibtex")
  },
  {
    "blueprint",
    FOUNDRY_STRV_INIT ("*.blp"),
    FOUNDRY_STRV_INIT ("text/x-blueprint")
  },
  {
    "bluespec",
    FOUNDRY_STRV_INIT ("*.bsv"),
    NULL
  },
  {
    "boo",
    FOUNDRY_STRV_INIT ("*.boo"),
    FOUNDRY_STRV_INIT ("text/x-boo")
  },
  {
    "c",
    FOUNDRY_STRV_INIT ("*.c"),
    FOUNDRY_STRV_INIT ("image/x-xpixmap", "text/x-c", "text/x-csrc")
  },
  {
    "cg",
    FOUNDRY_STRV_INIT ("*.cg"),
    NULL
  },
  {
    "changelog",
    FOUNDRY_STRV_INIT ("ChangeLog*"),
    FOUNDRY_STRV_INIT ("text/x-changelog")
  },
  {
    "chdr",
    FOUNDRY_STRV_INIT ("*.h"),
    FOUNDRY_STRV_INIT ("text/x-chdr")
  },
  {
    "cmake",
    FOUNDRY_STRV_INIT ("*.cmake", "*.cmake.in", "*.ctest", "*.ctest.in", "CMakeLists.txt"),
    NULL
  },
  {
    "cobol",
    FOUNDRY_STRV_INIT ("*.cbd", "*.cbl", "*.cdb", "*.cdc", "*.cob"),
    NULL
  },
  {
    "commonlisp",
    FOUNDRY_STRV_INIT ("*.asd", "*.lisp"),
    FOUNDRY_STRV_INIT ("text/x-common-lisp")
  },
  {
    "cpp",
    FOUNDRY_STRV_INIT ("*.C", "*.c++", "*.cc", "*.cpp", "*.cxx", "*.tpp"),
    FOUNDRY_STRV_INIT ("text/x-c++", "text/x-c++src", "text/x-cpp")
  },
  {
    "cpphdr",
    FOUNDRY_STRV_INIT ("*.h++", "*.hh", "*.hp", "*.hpp"),
    FOUNDRY_STRV_INIT ("text/x-c++hdr")
  },
  {
    "c-sharp",
    FOUNDRY_STRV_INIT ("*.cs"),
    FOUNDRY_STRV_INIT ("text/x-csharp", "text/x-csharpsrc")
  },
  {
    "css",
    FOUNDRY_STRV_INIT ("*.CSSL", "*.css"),
    FOUNDRY_STRV_INIT ("text/css")
  },
  {
    "csv",
    FOUNDRY_STRV_INIT ("*.csv"),
    FOUNDRY_STRV_INIT ("text/csv")
  },
  {
    "cuda",
    FOUNDRY_STRV_INIT ("*.cu", "*.cuh"),
    NULL
  },
  {
    "d",
    FOUNDRY_STRV_INIT ("*.d"),
    FOUNDRY_STRV_INIT ("text/x-dsrc")
  },
  {
    "dart",
    FOUNDRY_STRV_INIT ("*.dart"),
    FOUNDRY_STRV_INIT ("application/dart", "application/x-dart", "text/dart", "text/x-dart")
  },
  {
    "desktop",
    FOUNDRY_STRV_INIT ("*.desktop", "*.kdelnk"),
    FOUNDRY_STRV_INIT ("application/x-desktop", "application/x-gnome-app-info")
  },
  {
    "diff",
    FOUNDRY_STRV_INIT ("*.diff", "*.patch", "*.rej"),
    FOUNDRY_STRV_INIT ("text/x-diff", "text/x-patch", "text/x-reject")
  },
  {
    "docbook",
    FOUNDRY_STRV_INIT ("*.docbook"),
    FOUNDRY_STRV_INIT ("application/docbook+xml")
  },
  {
    "docker",
    FOUNDRY_STRV_INIT ("Containerfile", "Dockerfile"),
    FOUNDRY_STRV_INIT ("application/docker", "text/docker")
  },
  {
    "dosbatch",
    FOUNDRY_STRV_INIT ("*.bat", "*.cmd", "*.sys"),
    NULL
  },
  {
    "dot",
    FOUNDRY_STRV_INIT ("*.dot", "*.gv"),
    FOUNDRY_STRV_INIT ("text/vnd.graphviz")
  },
  {
    "dpatch",
    FOUNDRY_STRV_INIT ("*.dpatch"),
    FOUNDRY_STRV_INIT ("text/x-dpatch")
  },
  {
    "dtd",
    FOUNDRY_STRV_INIT ("*.dtd"),
    FOUNDRY_STRV_INIT ("text/x-dtd")
  },
  {
    "dtl",
    FOUNDRY_STRV_INIT ("*.dtl"),
    NULL
  },
  {
    "eiffel",
    FOUNDRY_STRV_INIT ("*.e", "*.eif"),
    FOUNDRY_STRV_INIT ("text/x-eiffel")
  },
  {
    "elixir",
    FOUNDRY_STRV_INIT ("*.ex", "*.exs"),
    FOUNDRY_STRV_INIT ("text/x-elixir")
  },
  {
    "erb-html",
    FOUNDRY_STRV_INIT ("*.html.erb", "*.rhtml"),
    FOUNDRY_STRV_INIT ("text/rhtml")
  },
  {
    "erb-js",
    FOUNDRY_STRV_INIT ("*.js.erb"),
    NULL
  },
  {
    "erb",
    FOUNDRY_STRV_INIT ("*.erb"),
    FOUNDRY_STRV_INIT ("text/erb")
  },
  {
    "erlang",
    FOUNDRY_STRV_INIT ("*.erl", "*.hrl"),
    FOUNDRY_STRV_INIT ("text/x-erlang")
  },
  {
    "fcl",
    FOUNDRY_STRV_INIT ("*.fcl"),
    NULL
  },
  {
    "fish",
    FOUNDRY_STRV_INIT ("*.fish"),
    FOUNDRY_STRV_INIT ("text/x-fish")
  },
  {
    "forth",
    FOUNDRY_STRV_INIT ("*.frt", "*.fs"),
    FOUNDRY_STRV_INIT ("text/x-forth")
  },
  {
    "fortran",
    FOUNDRY_STRV_INIT ("*.F", "*.F90", "*.f", "*.f90", "*.f95", "*.for"),
    FOUNDRY_STRV_INIT ("text/x-fortran")
  },
  {
    "fsharp",
    FOUNDRY_STRV_INIT ("*.fs"),
    FOUNDRY_STRV_INIT ("text/x-fsharp")
  },
  {
    "ftl",
    FOUNDRY_STRV_INIT ("*.ftl"),
    FOUNDRY_STRV_INIT ("text/x-fluent")
  },
  {
    "gap",
    FOUNDRY_STRV_INIT ("*.g", "*.gap", "*.gi"),
    FOUNDRY_STRV_INIT ("text/x-gap")
  },
  {
    "gdb-log",
    FOUNDRY_STRV_INIT ("*.gdb"),
    NULL
  },
  {
    "gdscript",
    FOUNDRY_STRV_INIT ("*.gd"),
    FOUNDRY_STRV_INIT ("text/x-gdscript")
  },
  {
    "genie",
    FOUNDRY_STRV_INIT ("*.gs"),
    FOUNDRY_STRV_INIT ("text/x-genie")
  },
  {
    "glsl",
    FOUNDRY_STRV_INIT ("*.glsl", "*.glslf", "*.glslv"),
    NULL
  },
  {
    "go",
    FOUNDRY_STRV_INIT ("*.go"),
    NULL
  },
  {
    "gradle",
    FOUNDRY_STRV_INIT ("*.gradle"),
    NULL
  },
  {
    "groff",
    FOUNDRY_STRV_INIT ("*.groff", "*.man"),
    FOUNDRY_STRV_INIT ("application/x-troff", "application/x-troff-man", "text/troff")
  },
  {
    "groovy",
    FOUNDRY_STRV_INIT ("*.groovy"),
    FOUNDRY_STRV_INIT ("text/x-groovy")
  },
  {
    "gtkrc",
    FOUNDRY_STRV_INIT (".gtkrc", ".gtkrc-*", "gtkrc", "gtkrc-*"),
    FOUNDRY_STRV_INIT ("text/x-gtkrc")
  },
  {
    "haskell-literate",
    FOUNDRY_STRV_INIT ("*.lhs"),
    FOUNDRY_STRV_INIT ("text/x-literate-haskell")
  },
  {
    "haskell",
    FOUNDRY_STRV_INIT ("*.hs"),
    FOUNDRY_STRV_INIT ("text/x-haskell")
  },
  {
    "haxe",
    FOUNDRY_STRV_INIT ("*.hx"),
    FOUNDRY_STRV_INIT ("text/x-haxe")
  },
  {
    "html",
    FOUNDRY_STRV_INIT ("*.htm", "*.html"),
    FOUNDRY_STRV_INIT ("text/html")
  },
  {
    "idl-exelis",
    FOUNDRY_STRV_INIT ("*.pro"),
    NULL
  },
  {
    "idl",
    FOUNDRY_STRV_INIT ("*.idl"),
    FOUNDRY_STRV_INIT ("text/x-idl")
  },
  {
    "imagej",
    FOUNDRY_STRV_INIT ("*.ijm"),
    NULL
  },
  {
    "ini",
    FOUNDRY_STRV_INIT ("*.ini"),
    FOUNDRY_STRV_INIT ("application/x-ini-file", "text/x-dbus-service", "text/x-ini-file", "text/x-systemd-unit")
  },
  {
    "j",
    FOUNDRY_STRV_INIT ("*.ijs"),
    NULL
  },
  {
    "jade",
    FOUNDRY_STRV_INIT ("*.jade", "*.pug"),
    NULL
  },
  {
    "java",
    FOUNDRY_STRV_INIT ("*.java"),
    FOUNDRY_STRV_INIT ("text/x-java")
  },
  {
    "js",
    FOUNDRY_STRV_INIT ("*.js", "*.mjs"),
    FOUNDRY_STRV_INIT ("application/javascript", "application/x-javascript", "text/javascript", "text/x-javascript", "text/x-js")
  },
  {
    "json",
    FOUNDRY_STRV_INIT ("*.geojson", "*.json", "*.topojson"),
    FOUNDRY_STRV_INIT ("application/json")
  },
  {
    "jsx",
    FOUNDRY_STRV_INIT ("*.jsx"),
    FOUNDRY_STRV_INIT ("application/jsx", "application/x-jsx", "text/jsx", "text/x-jsx")
  },
  {
    "julia",
    FOUNDRY_STRV_INIT ("*.jl"),
    NULL
  },
  {
    "kotlin",
    FOUNDRY_STRV_INIT ("*.kt", "*.kts"),
    FOUNDRY_STRV_INIT ("text/x-kotlin")
  },
  {
    "latex",
    FOUNDRY_STRV_INIT ("*.bbl", "*.cls", "*.dtx", "*.ins", "*.ltx", "*.sty", "*.tex"),
    FOUNDRY_STRV_INIT ("text/x-tex")
  },
  {
    "lean",
    FOUNDRY_STRV_INIT ("*.lean"),
    FOUNDRY_STRV_INIT ("text/x-lean")
  },
  {
    "less",
    FOUNDRY_STRV_INIT ("*.less"),
    FOUNDRY_STRV_INIT ("text/less", "text/x-less")
  },
  {
    "lex",
    FOUNDRY_STRV_INIT ("*.flex", "*.l", "*.lex"),
    NULL
  },
  {
    "libtool",
    FOUNDRY_STRV_INIT ("*.la", "*.lai", "*.lo"),
    FOUNDRY_STRV_INIT ("text/x-libtool")
  },
  {
    "llvm",
    FOUNDRY_STRV_INIT ("*.ll"),
    NULL
  },
  {
    "logcat",
    FOUNDRY_STRV_INIT ("*.logcat"),
    FOUNDRY_STRV_INIT ("text/x-logcat")
  },
  {
    "logtalk",
    FOUNDRY_STRV_INIT ("*.lgt"),
    FOUNDRY_STRV_INIT ("text/x-logtalk")
  },
  {
    "lua",
    FOUNDRY_STRV_INIT ("*.lua"),
    FOUNDRY_STRV_INIT ("text/x-lua")
  },
  {
    "m4",
    FOUNDRY_STRV_INIT ("*.m4", "configure.ac", "configure.in"),
    FOUNDRY_STRV_INIT ("application/x-m4")
  },
  {
    "makefile",
    FOUNDRY_STRV_INIT ("*.mak", "*.make", "*.mk", "GNUmakefile", "[Mm]akefile"),
    FOUNDRY_STRV_INIT ("text/x-makefile")
  },
  {
    "mallard",
    FOUNDRY_STRV_INIT ("*.page"),
    NULL
  },
  {
    "markdown",
    FOUNDRY_STRV_INIT ("*.markdown", "*.md", "*.mkd"),
    FOUNDRY_STRV_INIT ("text/x-markdown")
  },
  {
    "matlab",
    FOUNDRY_STRV_INIT ("*.m"),
    FOUNDRY_STRV_INIT ("text/x-matlab")
  },
  {
    "maxima",
    FOUNDRY_STRV_INIT ("*.DEM", "*.MAC", "*.WXM", "*.dem", "*.mac", "*.wxm"),
    FOUNDRY_STRV_INIT ("text/mxm")
  },
  {
    "meson",
    FOUNDRY_STRV_INIT ("meson.build", "meson.options", "meson_options.txt"),
    FOUNDRY_STRV_INIT ("text/x-meson")
  },
  {
    "modelica",
    FOUNDRY_STRV_INIT ("*.mo", "*.mop"),
    FOUNDRY_STRV_INIT ("text/x-modelica")
  },
  {
    "mxml",
    FOUNDRY_STRV_INIT ("*.mxml"),
    NULL
  },
  {
    "nemerle",
    FOUNDRY_STRV_INIT ("*.n"),
    FOUNDRY_STRV_INIT ("text/x-nemerle")
  },
  {
    "netrexx",
    FOUNDRY_STRV_INIT ("*.nrx"),
    FOUNDRY_STRV_INIT ("text/x-netrexx")
  },
  {
    "nix",
    FOUNDRY_STRV_INIT ("*.nix"),
    NULL
  },
  {
    "nsis",
    FOUNDRY_STRV_INIT ("*.nsh", "*.nsi"),
    NULL
  },
  {
    "objc",
    FOUNDRY_STRV_INIT ("*.m"),
    FOUNDRY_STRV_INIT ("text/x-objcsrc")
  },
  {
    "objj",
    FOUNDRY_STRV_INIT ("*.j"),
    FOUNDRY_STRV_INIT ("text/x-objective-j")
  },
  {
    "ocaml",
    FOUNDRY_STRV_INIT ("*.ml", "*.mli", "*.mll", "*.mly"),
    FOUNDRY_STRV_INIT ("text/x-ocaml")
  },
  {
    "ocl",
    FOUNDRY_STRV_INIT ("*.ocl"),
    FOUNDRY_STRV_INIT ("text/x-ocl")
  },
  {
    "octave",
    FOUNDRY_STRV_INIT ("*.m"),
    FOUNDRY_STRV_INIT ("text/x-octave")
  },
  {
    "ooc",
    FOUNDRY_STRV_INIT ("*.ooc"),
    NULL
  },
  {
    "opal",
    FOUNDRY_STRV_INIT ("*.impl", "*.sign"),
    NULL
  },
  {
    "opencl",
    FOUNDRY_STRV_INIT ("*.cl"),
    NULL
  },
  {
    "pascal",
    FOUNDRY_STRV_INIT ("*.p", "*.pas"),
    FOUNDRY_STRV_INIT ("text/x-pascal")
  },
  {
    "perl",
    FOUNDRY_STRV_INIT ("*.al", "*.perl", "*.pl", "*.pm", "*.t"),
    FOUNDRY_STRV_INIT ("application/x-perl", "text/x-perl")
  },
  {
    "php",
    FOUNDRY_STRV_INIT ("*.php", "*.php3", "*.php4", "*.phtml"),
    FOUNDRY_STRV_INIT ("application/x-php", "application/x-php-source", "text/x-php", "text/x-php-source")
  },
  {
    "pig",
    FOUNDRY_STRV_INIT ("*.pig"),
    NULL
  },
  {
    "pkgconfig",
    FOUNDRY_STRV_INIT ("*.pc"),
    FOUNDRY_STRV_INIT ("text/x-pkg-config")
  },
  {
    "gettext-translation",
    FOUNDRY_STRV_INIT ("*.po", "*.pot"),
    FOUNDRY_STRV_INIT ("text/x-gettext-translation", "text/x-gettext-translation-template", "text/x-po", "text/x-pot", "text/x-pox")
  },
  {
    "powershell",
    FOUNDRY_STRV_INIT ("*.ps1", "*.psd1", "*.psm1"),
    FOUNDRY_STRV_INIT ("text/x-powershell", "text/x-ps")
  },
  {
    "prolog",
    FOUNDRY_STRV_INIT ("*.prolog"),
    FOUNDRY_STRV_INIT ("text/x-prolog")
  },
  {
    "proto",
    FOUNDRY_STRV_INIT ("*.proto"),
    FOUNDRY_STRV_INIT ("text/x-protobuf")
  },
  {
    "puppet",
    FOUNDRY_STRV_INIT ("*.pp"),
    NULL
  },
  {
    "python",
    FOUNDRY_STRV_INIT ("*.py", "*.py2"),
    FOUNDRY_STRV_INIT ("application/x-python", "text/x-python")
  },
  {
    "python3",
    FOUNDRY_STRV_INIT ("*.py", "*.py3", "*.pyi"),
    FOUNDRY_STRV_INIT ("application/x-python", "text/x-python", "text/x-python3")
  },
  {
    "reasonml",
    FOUNDRY_STRV_INIT ("*.re", "*.rei"),
    NULL
  },
  {
    "rpmspec",
    FOUNDRY_STRV_INIT ("*.spec"),
    FOUNDRY_STRV_INIT ("text/x-rpm-spec")
  },
  {
    "rst",
    FOUNDRY_STRV_INIT ("*.rst"),
    FOUNDRY_STRV_INIT ("text/x-rst")
  },
  {
    "ruby",
    FOUNDRY_STRV_INIT ("*.gemspec", "*.rake", "*.rb", "Capfile", "Gemfile", "Rakefile"),
    FOUNDRY_STRV_INIT ("application/x-ruby", "text/x-ruby")
  },
  {
    "rust",
    FOUNDRY_STRV_INIT ("*.rs"),
    FOUNDRY_STRV_INIT ("text/rust")
  },
  {
    "scala",
    FOUNDRY_STRV_INIT ("*.scala"),
    FOUNDRY_STRV_INIT ("text/x-scala")
  },
  {
    "scheme",
    FOUNDRY_STRV_INIT ("*.scm"),
    FOUNDRY_STRV_INIT ("text/x-scheme")
  },
  {
    "scilab",
    FOUNDRY_STRV_INIT ("*.sce", "*.sci"),
    NULL
  },
  {
    "scss",
    FOUNDRY_STRV_INIT ("*.scss"),
    FOUNDRY_STRV_INIT ("text/x-scss")
  },
  {
    "sh",
    FOUNDRY_STRV_INIT ("*.sh", "*bashrc", ".bash_login", ".bash_logout", ".bash_profile", ".profile"),
    FOUNDRY_STRV_INIT ("application/x-shellscript", "text/x-sh", "text/x-shellscript")
  },
  {
    "sml",
    FOUNDRY_STRV_INIT ("*.sig", "*.sml"),
    NULL
  },
  {
    "solidity",
    FOUNDRY_STRV_INIT ("*.sol", "*.solidity"),
    NULL
  },
  {
    "sparql",
    FOUNDRY_STRV_INIT ("*.rq"),
    FOUNDRY_STRV_INIT ("application/sparql-query")
  },
  {
    "spice",
    FOUNDRY_STRV_INIT ("*.CIR", "*.Cir", "*.cir"),
    NULL
  },
  {
    "sql",
    FOUNDRY_STRV_INIT ("*.sql"),
    FOUNDRY_STRV_INIT ("text/x-sql")
  },
  {
    "star",
    FOUNDRY_STRV_INIT ("*.cif", "*.mif", "*.str"),
    NULL
  },
  {
    "sweave",
    FOUNDRY_STRV_INIT ("*.Rnw", "*.Snw", "*.rnw", "*.snw"),
    NULL
  },
  {
    "swift",
    FOUNDRY_STRV_INIT ("*.swift"),
    FOUNDRY_STRV_INIT ("text/x-swift")
  },
  {
    "systemverilog",
    FOUNDRY_STRV_INIT ("*.sv", "*.svh"),
    NULL
  },
  {
    "t2t",
    FOUNDRY_STRV_INIT ("*.t2t"),
    NULL
  },
  {
    "tcl",
    FOUNDRY_STRV_INIT ("*.tcl", "*.tk"),
    FOUNDRY_STRV_INIT ("application/x-tcl", "text/x-tcl")
  },
  {
    "tera",
    FOUNDRY_STRV_INIT ("*.tera"),
    NULL
  },
  {
    "terraform",
    FOUNDRY_STRV_INIT ("*.hcl", "*.tf", "*.tfvars"),
    NULL
  },
  {
    "texinfo",
    FOUNDRY_STRV_INIT ("*.texi", "*.texinfo"),
    FOUNDRY_STRV_INIT ("text/x-texinfo")
  },
  {
    "thrift",
    FOUNDRY_STRV_INIT ("*.thrift"),
    NULL
  },
  {
    "toml",
    FOUNDRY_STRV_INIT ("*.lock", "*.tml", "*.toml"),
    NULL
  },
  {
    "twig",
    FOUNDRY_STRV_INIT ("*.twig"),
    FOUNDRY_STRV_INIT ("text/x-twig")
  },
  {
    "typescript-jsx",
    FOUNDRY_STRV_INIT ("*.tsx"),
    FOUNDRY_STRV_INIT ("application/typescript-jsx", "application/x-typescript-jsx", "text/typescript-jsx", "text/x-typescript-jsx")
  },
  {
    "typescript",
    FOUNDRY_STRV_INIT ("*.ts"),
    FOUNDRY_STRV_INIT ("application/typescript", "application/x-typescript", "text/typescript", "text/x-typescript")
  },
  {
    "vala",
    FOUNDRY_STRV_INIT ("*.vala", "*.vapi"),
    FOUNDRY_STRV_INIT ("text/x-vala")
  },
  {
    "vbnet",
    FOUNDRY_STRV_INIT ("*.vb"),
    FOUNDRY_STRV_INIT ("text/x-vb", "text/x-vbnet")
  },
  {
    "verilog",
    FOUNDRY_STRV_INIT ("*.v"),
    FOUNDRY_STRV_INIT ("text/x-verilog-src")
  },
  {
    "vhdl",
    FOUNDRY_STRV_INIT ("*.vhd"),
    FOUNDRY_STRV_INIT ("text/x-vhdl")
  },
  {
    "wren",
    FOUNDRY_STRV_INIT ("*.wren"),
    FOUNDRY_STRV_INIT ("application/x-wren")
  },
  {
    "xml",
    FOUNDRY_STRV_INIT ("*.abw", "*.fo", "*.glade", "*.jnlp", "*.kino", "*.lang", "*.mml", "*.rdf", "*.rss", "*.sgml", "*.siv", "*.smi", "*.smil", "*.sml", "*.svg", "*.wml", "*.xbel", "*.xhtml", "*.xmi", "*.xml", "*.xslfo", "*.xspf", "*.xul", "*.zabw"),
    FOUNDRY_STRV_INIT ("application/xml", "text/sgml", "text/xml")
  },
  {
    "xslt",
    FOUNDRY_STRV_INIT ("*.xsl", "*.xslt"),
    FOUNDRY_STRV_INIT ("application/xslt+xml")
  },
  {
    "yacc",
    FOUNDRY_STRV_INIT ("*.y", "*.yacc"),
    FOUNDRY_STRV_INIT ("text/x-bison", "text/x-yacc")
  },
  {
    "yaml",
    FOUNDRY_STRV_INIT ("*.yaml", "*.yml"),
    FOUNDRY_STRV_INIT ("application/x-yaml")
  },
  {
    "yara",
    FOUNDRY_STRV_INIT ("*.yar", "*.yara"),
    FOUNDRY_STRV_INIT ("text/x-yara", "text/yara")
  },
  {
    "zig",
    FOUNDRY_STRV_INIT ("*.zig", "*.zon"),
    FOUNDRY_STRV_INIT ("text/x-zig")
  },
};


G_END_DECLS
