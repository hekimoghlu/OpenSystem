/* plugin-doap-file.c
 *
 * Copyright 2015-2019 Christian Hergert <chergert@redhat.com>
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

#include "config.h"

#include <glib/gi18n-lib.h>

#include "plugin-doap-file.h"
#include "xml-reader-private.h"

/* TODO: We don't do any XMLNS checking or anything here. */

struct _PluginDoapFile
{
  GObject    parent_instance;

  char      *bug_database;
  char      *category;
  char      *description;
  char      *download_page;
  char      *homepage;;
  char      *name;
  char      *shortdesc;

  GPtrArray *languages;
  GList     *maintainers;
};

G_DEFINE_QUARK (plugin_doap_file_error, plugin_doap_file_error)
G_DEFINE_FINAL_TYPE (PluginDoapFile, plugin_doap_file, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_BUG_DATABASE,
  PROP_CATEGORY,
  PROP_DESCRIPTION,
  PROP_DOWNLOAD_PAGE,
  PROP_HOMEPAGE,
  PROP_LANGUAGES,
  PROP_NAME,
  PROP_SHORTDESC,
  N_PROPS
};

static GParamSpec *properties [N_PROPS];

PluginDoapFile *
plugin_doap_file_new (void)
{
  return g_object_new (PLUGIN_TYPE_DOAP_FILE, NULL);
}

const char *
plugin_doap_file_get_name (PluginDoapFile *self)
{
  g_return_val_if_fail (PLUGIN_IS_DOAP_FILE (self), NULL);

  return self->name;
}

const char *
plugin_doap_file_get_shortdesc (PluginDoapFile *self)
{
  g_return_val_if_fail (PLUGIN_IS_DOAP_FILE (self), NULL);

  return self->shortdesc;
}

const char *
plugin_doap_file_get_description (PluginDoapFile *self)
{
  g_return_val_if_fail (PLUGIN_IS_DOAP_FILE (self), NULL);

  return self->description;
}

const char *
plugin_doap_file_get_bug_database (PluginDoapFile *self)
{
  g_return_val_if_fail (PLUGIN_IS_DOAP_FILE (self), NULL);

  return self->bug_database;
}

const char *
plugin_doap_file_get_download_page (PluginDoapFile *self)
{
  g_return_val_if_fail (PLUGIN_IS_DOAP_FILE (self), NULL);

  return self->download_page;
}

const char *
plugin_doap_file_get_homepage (PluginDoapFile *self)
{
  g_return_val_if_fail (PLUGIN_IS_DOAP_FILE (self), NULL);

  return self->homepage;
}

const char *
plugin_doap_file_get_category (PluginDoapFile *self)
{
  g_return_val_if_fail (PLUGIN_IS_DOAP_FILE (self), NULL);

  return self->category;
}

/**
 * plugin_doap_file_get_languages:
 *
 * Returns: (transfer none): a #GStrv.
 */
char **
plugin_doap_file_get_languages (PluginDoapFile *self)
{
  g_return_val_if_fail (PLUGIN_IS_DOAP_FILE (self), NULL);

  if (self->languages != NULL)
    return (char **)self->languages->pdata;

  return NULL;
}

static void
plugin_doap_file_set_bug_database (PluginDoapFile *self,
                                   const char     *bug_database)
{
  g_return_if_fail (PLUGIN_IS_DOAP_FILE (self));

  if (g_set_str (&self->bug_database, bug_database))
    {
      g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_BUG_DATABASE]);
    }
}

static void
plugin_doap_file_set_category (PluginDoapFile *self,
                               const char     *category)
{
  g_return_if_fail (PLUGIN_IS_DOAP_FILE (self));

  if (g_set_str (&self->category, category))
    {
      g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_CATEGORY]);
    }
}

static void
plugin_doap_file_set_description (PluginDoapFile *self,
                                  const char     *description)
{
  g_return_if_fail (PLUGIN_IS_DOAP_FILE (self));

  if (g_set_str (&self->description, description))
    {
      g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_DESCRIPTION]);
    }
}

static void
plugin_doap_file_set_download_page (PluginDoapFile *self,
                                    const char     *download_page)
{
  g_return_if_fail (PLUGIN_IS_DOAP_FILE (self));

  if (g_set_str (&self->download_page, download_page))
    {
      g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_DOWNLOAD_PAGE]);
    }
}

static void
plugin_doap_file_set_homepage (PluginDoapFile *self,
                               const char     *homepage)
{
  g_return_if_fail (PLUGIN_IS_DOAP_FILE (self));

  if (g_set_str (&self->homepage, homepage))
    {
      g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_HOMEPAGE]);
    }
}

static void
plugin_doap_file_set_name (PluginDoapFile *self,
                           const char     *name)
{
  g_return_if_fail (PLUGIN_IS_DOAP_FILE (self));

  if (g_set_str (&self->name, name))
    {
      g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_NAME]);
    }
}

static void
plugin_doap_file_set_shortdesc (PluginDoapFile *self,
                                const char     *shortdesc)
{
  g_autofree char *tmp_str = NULL;

  g_return_if_fail (PLUGIN_IS_DOAP_FILE (self));

  tmp_str = g_strdelimit (g_strdup (shortdesc), "\n", ' ');
  if (g_set_str (&self->shortdesc, tmp_str))
    {
      g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_SHORTDESC]);
    }
}

/**
 * plugin_doap_file_get_maintainers:
 *
 *
 *
 * Returns: (transfer none) (element-type PluginDoapPerson): a #GList of #PluginDoapPerson.
 */
GList *
plugin_doap_file_get_maintainers (PluginDoapFile *self)
{
  g_return_val_if_fail (PLUGIN_IS_DOAP_FILE (self), NULL);

  return self->maintainers;
}

static void
plugin_doap_file_add_language (PluginDoapFile *self,
                               const char     *language)
{
  g_return_if_fail (PLUGIN_IS_DOAP_FILE (self));
  g_return_if_fail (language != NULL);

  if (self->languages == NULL)
    {
      self->languages = g_ptr_array_new_with_free_func (g_free);
      g_ptr_array_add (self->languages, NULL);
    }

  g_assert (self->languages->len > 0);

  g_ptr_array_index (self->languages, self->languages->len - 1) = g_strdup (language);
  g_ptr_array_add (self->languages, NULL);

  g_object_notify_by_pspec (G_OBJECT (self), properties [PROP_LANGUAGES]);
}

static void
plugin_doap_file_set_languages (PluginDoapFile  *self,
                                char           **languages)
{
  gsize i;

  g_return_if_fail (PLUGIN_IS_DOAP_FILE (self));

  if ((self->languages != NULL) && (self->languages->len > 0))
    g_ptr_array_remove_range (self->languages, 0, self->languages->len);

  g_object_freeze_notify (G_OBJECT (self));
  for (i = 0; languages [i]; i++)
    plugin_doap_file_add_language (self, languages [i]);
  g_object_thaw_notify (G_OBJECT (self));
}

static void
plugin_doap_file_finalize (GObject *object)
{
  PluginDoapFile *self = (PluginDoapFile *)object;

  g_clear_pointer (&self->bug_database, g_free);
  g_clear_pointer (&self->category, g_free);
  g_clear_pointer (&self->description, g_free);
  g_clear_pointer (&self->download_page, g_free);
  g_clear_pointer (&self->homepage, g_free);
  g_clear_pointer (&self->languages, g_ptr_array_unref);
  g_clear_pointer (&self->name, g_free);
  g_clear_pointer (&self->shortdesc, g_free);

  g_list_free_full (self->maintainers, g_object_unref);
  self->maintainers = NULL;

  G_OBJECT_CLASS (plugin_doap_file_parent_class)->finalize (object);
}

static void
plugin_doap_file_get_property (GObject    *object,
                               guint       prop_id,
                               GValue     *value,
                               GParamSpec *pspec)
{
  PluginDoapFile *self = PLUGIN_DOAP_FILE (object);

  switch (prop_id)
    {
    case PROP_BUG_DATABASE:
      g_value_set_string (value, plugin_doap_file_get_bug_database (self));
      break;

    case PROP_CATEGORY:
      g_value_set_string (value, plugin_doap_file_get_category (self));
      break;

    case PROP_DESCRIPTION:
      g_value_set_string (value, plugin_doap_file_get_description (self));
      break;

    case PROP_DOWNLOAD_PAGE:
      g_value_set_string (value, plugin_doap_file_get_download_page (self));
      break;

    case PROP_HOMEPAGE:
      g_value_set_string (value, plugin_doap_file_get_homepage (self));
      break;

    case PROP_LANGUAGES:
      g_value_set_boxed (value, plugin_doap_file_get_languages (self));
      break;

    case PROP_NAME:
      g_value_set_string (value, plugin_doap_file_get_name (self));
      break;

    case PROP_SHORTDESC:
      g_value_set_string (value, plugin_doap_file_get_shortdesc (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_doap_file_set_property (GObject      *object,
                               guint         prop_id,
                               const GValue *value,
                               GParamSpec   *pspec)
{
  PluginDoapFile *self = PLUGIN_DOAP_FILE (object);

  switch (prop_id)
    {
    case PROP_BUG_DATABASE:
      plugin_doap_file_set_bug_database (self, g_value_get_string (value));
      break;

    case PROP_CATEGORY:
      plugin_doap_file_set_category (self, g_value_get_string (value));
      break;

    case PROP_DESCRIPTION:
      plugin_doap_file_set_description (self, g_value_get_string (value));
      break;

    case PROP_DOWNLOAD_PAGE:
      plugin_doap_file_set_download_page (self, g_value_get_string (value));
      break;

    case PROP_HOMEPAGE:
      plugin_doap_file_set_homepage (self, g_value_get_string (value));
      break;

    case PROP_LANGUAGES:
      plugin_doap_file_set_languages (self, g_value_get_boxed (value));
      break;

    case PROP_NAME:
      plugin_doap_file_set_name (self, g_value_get_string (value));
      break;

    case PROP_SHORTDESC:
      plugin_doap_file_set_shortdesc (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
plugin_doap_file_class_init (PluginDoapFileClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = plugin_doap_file_finalize;
  object_class->get_property = plugin_doap_file_get_property;
  object_class->set_property = plugin_doap_file_set_property;

  properties [PROP_BUG_DATABASE] =
    g_param_spec_string ("bug-database",
                         "Bug Database",
                         "Bug Database",
                         NULL,
                         (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  properties [PROP_CATEGORY] =
    g_param_spec_string ("category",
                         "Category",
                         "Category",
                         NULL,
                         (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  properties [PROP_DESCRIPTION] =
    g_param_spec_string ("description",
                         "Description",
                         "Description",
                         NULL,
                         (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  properties [PROP_DOWNLOAD_PAGE] =
    g_param_spec_string ("download-page",
                         "Download Page",
                         "Download Page",
                         NULL,
                         (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  properties [PROP_HOMEPAGE] =
    g_param_spec_string ("homepage",
                         "Homepage",
                         "Homepage",
                         NULL,
                         (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  properties [PROP_LANGUAGES] =
    g_param_spec_string ("languages",
                         "Languages",
                         "Languages",
                         NULL,
                         (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  properties [PROP_NAME] =
    g_param_spec_string ("name",
                         "Name",
                         "Name",
                         NULL,
                         (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  properties [PROP_SHORTDESC] =
    g_param_spec_string ("shortdesc",
                         "Shortdesc",
                         "Shortdesc",
                         NULL,
                         (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
plugin_doap_file_init (PluginDoapFile *self)
{
}

static gboolean
plugin_doap_file_parse_maintainer (PluginDoapFile *self,
                                   XmlReader      *reader)
{
  g_assert (PLUGIN_IS_DOAP_FILE (self));
  g_assert (XML_IS_READER (reader));

  if (!xml_reader_read (reader))
    return FALSE;

  do
    {
      if (xml_reader_is_a_local (reader, "Person") && xml_reader_read (reader))
        {
          g_autoptr(PluginDoapPerson) person = plugin_doap_person_new ();

          do
            {
              if (xml_reader_is_a_local (reader, "name"))
                {
                  char *str;

                  str = xml_reader_read_string (reader);
                  plugin_doap_person_set_name (person, str);
                  g_free (str);
                }
              else if (xml_reader_is_a_local (reader, "mbox"))
                {
                  char *str;

                  str = xml_reader_get_attribute (reader, "rdf:resource");
                  if (str != NULL && str[0] != '\0' && g_str_has_prefix (str, "mailto:"))
                    plugin_doap_person_set_email (person, str + strlen ("mailto:"));
                  g_free (str);
                }
            }
          while (xml_reader_read_to_next (reader));

          if (plugin_doap_person_get_name (person) || plugin_doap_person_get_email (person))
            self->maintainers = g_list_append (self->maintainers, g_object_ref (person));
        }
    }
  while (xml_reader_read_to_next (reader));

  return TRUE;
}

static gboolean
load_doap (PluginDoapFile  *self,
           XmlReader       *reader,
           GError         **error)
{
  if (!xml_reader_read_start_element (reader, "Project"))
    {
      g_set_error (error,
                   PLUGIN_DOAP_FILE_ERROR,
                   PLUGIN_DOAP_FILE_ERROR_INVALID_FORMAT,
                   "Project element is missing from doap.");
      return FALSE;
    }

  g_object_freeze_notify (G_OBJECT (self));

  xml_reader_read (reader);

  do
    {
      const char *element_name;

      element_name = xml_reader_get_local_name (reader);

      if (g_strcmp0 (element_name, "name") == 0 ||
          g_strcmp0 (element_name, "shortdesc") == 0 ||
          g_strcmp0 (element_name, "description") == 0)
        {
          char *str;

          str = xml_reader_read_string (reader);
          if (str != NULL)
            g_object_set (self, element_name, g_strstrip (str), NULL);
          g_free (str);
        }
      else if (g_strcmp0 (element_name, "category") == 0 ||
               g_strcmp0 (element_name, "homepage") == 0 ||
               g_strcmp0 (element_name, "download-page") == 0 ||
               g_strcmp0 (element_name, "bug-database") == 0)
        {
          char *str;

          str = xml_reader_get_attribute (reader, "rdf:resource");
          if (str != NULL)
            g_object_set (self, element_name, g_strstrip (str), NULL);
          g_free (str);
        }
      else if (g_strcmp0 (element_name, "programming-language") == 0)
        {
          char *str;

          str = xml_reader_read_string (reader);
          if (str != NULL && str[0] != '\0')
            plugin_doap_file_add_language (self, g_strstrip (str));
          g_free (str);
        }
      else if (g_strcmp0 (element_name, "maintainer") == 0)
        {
          if (!plugin_doap_file_parse_maintainer (self, reader))
            break;
        }
    }
  while (xml_reader_read_to_next (reader));

  g_object_thaw_notify (G_OBJECT (self));

  return TRUE;
}

gboolean
plugin_doap_file_load_from_file (PluginDoapFile  *self,
                                 GFile           *file,
                                 GCancellable    *cancellable,
                                 GError         **error)
{
  g_autoptr(XmlReader) reader = NULL;

  g_return_val_if_fail (PLUGIN_IS_DOAP_FILE (self), FALSE);
  g_return_val_if_fail (G_IS_FILE (file), FALSE);
  g_return_val_if_fail (!cancellable || G_IS_CANCELLABLE (cancellable), FALSE);

  reader = xml_reader_new ();

  if (!xml_reader_load_from_file (reader, file, cancellable, error))
    return FALSE;

  return load_doap (self, reader, error);
}

gboolean
plugin_doap_file_load_from_data (PluginDoapFile  *self,
                                 const char      *data,
                                 gsize            length,
                                 GError         **error)
{
  g_autoptr(XmlReader) reader = NULL;

  g_return_val_if_fail (PLUGIN_IS_DOAP_FILE (self), FALSE);
  g_return_val_if_fail (data != NULL, FALSE);

  reader = xml_reader_new ();

  if (!xml_reader_load_from_data (reader, (const char *)data, length, NULL, NULL))
    return FALSE;

  return load_doap (self, reader, error);
}

gboolean
plugin_doap_file_load_from_bytes (PluginDoapFile  *self,
                                  GBytes          *bytes,
                                  GError         **error)
{
  g_return_val_if_fail (PLUGIN_IS_DOAP_FILE (self), FALSE);
  g_return_val_if_fail (bytes != NULL, FALSE);

  return plugin_doap_file_load_from_data (self,
                                          g_bytes_get_data (bytes, NULL),
                                          g_bytes_get_size (bytes),
                                          error);
}
