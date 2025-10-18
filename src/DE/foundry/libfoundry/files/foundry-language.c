/* foundry-language.c
 *
 * Copyright 2025 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "foundry-language.h"

struct _FoundryLanguage
{
  GObject parent_instance;
  char *id;
  char *name;
  char *meson_id;
};

G_DEFINE_FINAL_TYPE (FoundryLanguage, foundry_language, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_ID,
  PROP_MESON_ID,
  PROP_NAME,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_language_finalize (GObject *object)
{
  FoundryLanguage *self = (FoundryLanguage *)object;

  g_clear_pointer (&self->id, g_free);
  g_clear_pointer (&self->name, g_free);
  g_clear_pointer (&self->meson_id, g_free);

  G_OBJECT_CLASS (foundry_language_parent_class)->finalize (object);
}

static void
foundry_language_get_property (GObject    *object,
                               guint       prop_id,
                               GValue     *value,
                               GParamSpec *pspec)
{
  FoundryLanguage *self = FOUNDRY_LANGUAGE (object);

  switch (prop_id)
    {
    case PROP_ID:
      g_value_set_string (value, foundry_language_get_id (self));
      break;

    case PROP_MESON_ID:
      g_value_set_string (value, foundry_language_get_meson_id (self));
      break;

    case PROP_NAME:
      g_value_set_string (value, foundry_language_get_name (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_language_set_property (GObject      *object,
                               guint         prop_id,
                               const GValue *value,
                               GParamSpec   *pspec)
{
  FoundryLanguage *self = FOUNDRY_LANGUAGE (object);

  switch (prop_id)
    {
    case PROP_ID:
      self->id = g_value_dup_string (value);
      break;

    case PROP_MESON_ID:
      self->meson_id = g_value_dup_string (value);
      break;

    case PROP_NAME:
      self->name = g_value_dup_string (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_language_class_init (FoundryLanguageClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_language_finalize;
  object_class->get_property = foundry_language_get_property;
  object_class->set_property = foundry_language_set_property;

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_MESON_ID] =
    g_param_spec_string ("meson-id", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_language_init (FoundryLanguage *self)
{
}

FoundryLanguage *
foundry_language_new (const char *id,
                      const char *name,
                      const char *meson_id)
{
  g_return_val_if_fail (id != NULL, NULL);
  g_return_val_if_fail (name != NULL, NULL);

  return g_object_new (FOUNDRY_TYPE_LANGUAGE,
                       "id", id,
                       "name", name,
                       "meson-id", meson_id,
                       NULL);
}

const char *
foundry_language_get_id (FoundryLanguage *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LANGUAGE (self), NULL);

  return self->id;
}

/**
 * foundry_language_get_meson_id:
 * @self: a [class@Foundry.Language]
 *
 * Returns: (nullable):
 */
const char *
foundry_language_get_meson_id (FoundryLanguage *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LANGUAGE (self), NULL);

  return self->meson_id;
}

const char *
foundry_language_get_name (FoundryLanguage *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LANGUAGE (self), NULL);

  return self->name;
}

guint
foundry_language_hash (const FoundryLanguage *self)
{
  return g_str_hash (self->id);
}

gboolean
foundry_language_equal (const FoundryLanguage *self,
                        const FoundryLanguage *other)
{
  return g_str_equal (self->id, other->id);
}

static GHashTable *id_to_language;
static GListModel *all;

static const struct {
  const char *id;
  const char *name;
  const char *meson_id;
} languages[] = {
  /* TODO: Should we translate? */
  { "plain-text", "Plain Text" },
  { "abnf", "ABNF" },
  { "actionscript", "ActionScript" },
  { "ada", "Ada" },
  { "ansforth94", "ANS-Forth94" },
  { "asp", "ASP" },
  { "automake", "Automake" },
  { "awk", "awk" },
  { "bennugd", "BennuGD" },
  { "bibtex", "BibTeX" },
  { "bluespec", "Bluespec SystemVerilog" },
  { "boo", "Boo" },
  { "c", "C", "c" },
  { "c-sharp", "C#", "cs" },
  { "cpp", "C++", "cpp" },
  { "cg", "CG Shader Language" },
  { "changelog", "ChangeLog" },
  { "cpphdr", "C++ Header" },
  { "cmake", "CMake" },
  { "chdr", "C/ObjC Header" },
  { "cobol", "COBOL" },
  { "css", "CSS" },
  { "csv", "CSV" },
  { "cuda", "CUDA", "cuda" },
  { "d", "D", "d" },
  { "desktop", ".desktop" },
  { "diff", "Diff" },
  { "dtl", "Django Template" },
  { "docbook", "DocBook" },
  { "dosbatch", "DOS Batch" },
  { "dpatch", "DPatch" },
  { "dtd", "DTD" },
  { "eiffel", "Eiffel" },
  { "erlang", "Erlang" },
  { "fsharp", "F#" },
  { "fcl", "FCL" },
  { "forth", "Forth" },
  { "fortran", "Fortran 95", "fortran" },
  { "gap", "GAP" },
  { "gdb-log", "GDB Log" },
  { "genie", "Genie" },
  { "gettext-translation", "gettext translation" },
  { "go", "Go" },
  { "dot", "Graphviz Dot" },
  { "groovy", "Groovy" },
  { "gtk-doc", "gtk-doc" },
  { "gtkrc", "GtkRC" },
  { "haddock", "Haddock" },
  { "haskell", "Haskell" },
  { "haxe", "Haxe" },
  { "html", "HTML" },
  { "idl", "IDL" },
  { "idl-exelis", "IDL-Exelis" },
  { "imagej", "ImageJ" },
  { "ini", ".ini" },
  { "j", "J" },
  { "jade", "Jade" },
  { "java", "Java", "java" },
  { "js", "JavaScript" },
  { "json", "JSON" },
  { "julia", "Julia" },
  { "kotlin", "Kotlin" },
  { "latex", "LaTeX" },
  { "less", "Less" },
  { "lex", "Lex" },
  { "libtool", "libtool" },
  { "haskell-literate", "Literate Haskell" },
  { "llvm", "LLVM IR" },
  { "logcat", "logcat" },
  { "lua", "Lua" },
  { "m4", "m4" },
  { "makefile", "Makefile" },
  { "mallard", "Mallard" },
  { "markdown", "Markdown" },
  { "matlab", "Matlab" },
  { "maxima", "Maxima" },
  { "mediawiki", "MediaWiki" },
  { "meson", "Meson" },
  { "modelica", "Modelica" },
  { "mxml", "MXML" },
  { "nemerle", "Nemerle" },
  { "netrexx", "NetRexx" },
  { "nix", "Nix" },
  { "nsis", "NSIS" },
  { "objc", "Objective-C", "objc" },
  { "objj", "Objective-J" },
  { "ocaml", "OCaml" },
  { "ocl", "OCL" },
  { "octave", "Octave" },
  { "ooc", "OOC" },
  { "opal", "Opal" },
  { "opencl", "OpenCL" },
  { "glsl", "OpenGL Shading Language" },
  { "pascal", "Pascal" },
  { "perl", "Perl" },
  { "php", "PHP" },
  { "pig", "Pig" },
  { "pkgconfig", "pkg-config" },
  { "prolog", "Prolog" },
  { "proto", "Protobuf" },
  { "puppet", "Puppet" },
  { "python", "Python 2" },
  { "python3", "Python" },
  { "r", "R" },
  { "rst", "reStructuredText" },
  { "rpmspec", "RPM spec" },
  { "ruby", "Ruby" },
  { "rust", "Rust", "rust" },
  { "scala", "Scala" },
  { "scheme", "Scheme" },
  { "scilab", "Scilab" },
  { "scss", "SCSS" },
  { "sh", "sh" },
  { "sparql", "SPARQL" },
  { "sql", "SQL" },
  { "sml", "Standard ML" },
  { "sweave", "Sweave" },
  { "swift", "Swift" },
  { "systemverilog", "SystemVerilog" },
  { "tcl", "Tcl" },
  { "tera", "Tera Template" },
  { "texinfo", "Texinfo" },
  { "thrift", "Thrift" },
  { "toml", "TOML" },
  { "t2t", "txt2tags" },
  { "vala", "Vala", "vala" },
  { "vbnet", "VB.NET" },
  { "verilog", "Verilog" },
  { "vhdl", "VHDL" },
  { "xml", "XML" },
  { "xslt", "XSLT" },
  { "yacc", "Yacc" },
  { "yaml", "YAML" },
};

static void
foundry_language_init_all (void)
{
  /* TODO: We might want to defer this to a provider so that we can get
   * updated contents from GtkSourceView without having to manage a table
   * of information.
   */

  if (g_once_init_enter (&all))
    {
      GListStore *store = g_list_store_new (FOUNDRY_TYPE_LANGUAGE);

      id_to_language = g_hash_table_new_full (g_str_hash, g_str_equal, NULL, g_object_unref);

      for (guint i = 0; i < G_N_ELEMENTS (languages); i++)
        {
          FoundryLanguage *language;

          language = g_object_new (FOUNDRY_TYPE_LANGUAGE,
                                   "id", languages[i].id,
                                   "name", languages[i].name,
                                   "meson-id", languages[i].meson_id,
                                   NULL);
          g_list_store_append (store, language);
          g_hash_table_insert (id_to_language,
                               (char *)languages[i].id,
                               g_steal_pointer (&language));
        }

      g_once_init_leave (&all, G_LIST_MODEL (store));
    }
}

/**
 * foundry_language_list_all:
 *
 * Returns: (transfer full): a [iface@Gio.ListModel] of
 *   [class@Foundry.Language].
 */
GListModel *
foundry_language_list_all (void)
{
  foundry_language_init_all ();

  return g_object_ref (all);
}

/**
 * foundry_language_find:
 * @id: the id of the language
 *
 * Returns: (transfer full):
 */
FoundryLanguage *
foundry_language_find (const char *id)
{
  FoundryLanguage *ret;

  g_return_val_if_fail (id != NULL, NULL);

  foundry_language_init_all ();

  if ((ret = g_hash_table_lookup (id_to_language, id)))
    return g_object_ref (ret);

  return NULL;
}
