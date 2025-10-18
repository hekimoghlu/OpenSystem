/* foundry-license.c
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

#include "config.h"

#include <glib/gi18n-lib.h>

#include "foundry-license.h"

struct _FoundryLicense
{
  GObject parent_instance;
  char *id;
  char *title;
  GBytes *text;
  GBytes *snippet_text;
};

enum {
  PROP_0,
  PROP_ID,
  PROP_TITLE,
  PROP_TEXT,
  PROP_TEXT_BYTES,
  PROP_SNIPPET_TEXT,
  PROP_SNIPPET_TEXT_BYTES,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryLicense, foundry_license, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_license_finalize (GObject *object)
{
  FoundryLicense *self = (FoundryLicense *)object;

  g_clear_pointer (&self->id, g_free);
  g_clear_pointer (&self->title, g_free);
  g_clear_pointer (&self->text, g_free);
  g_clear_pointer (&self->snippet_text, g_free);

  G_OBJECT_CLASS (foundry_license_parent_class)->finalize (object);
}

static void
foundry_license_get_property (GObject    *object,
                              guint       prop_id,
                              GValue     *value,
                              GParamSpec *pspec)
{
  FoundryLicense *self = FOUNDRY_LICENSE (object);

  switch (prop_id)
    {
    case PROP_ID:
      g_value_take_string (value, foundry_license_dup_id (self));
      break;

    case PROP_SNIPPET_TEXT:
      if (self->snippet_text)
        {
          gsize size;
          const char *data;

          data = g_bytes_get_data (self->snippet_text, &size);
          g_value_take_string (value, g_strndup (data, size));
        }
      break;

    case PROP_TITLE:
      g_value_take_string (value, foundry_license_dup_title (self));
      break;

    case PROP_TEXT_BYTES:
      g_value_take_boxed (value, foundry_license_dup_text (self));
      break;

    case PROP_TEXT:
      if (self->text)
        {
          gsize size;
          const char *data;

          data = g_bytes_get_data (self->text, &size);
          g_value_take_string (value, g_strndup (data, size));
        }
      break;

    case PROP_SNIPPET_TEXT_BYTES:
      g_value_take_boxed (value, foundry_license_dup_snippet_text (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_license_set_property (GObject      *object,
                              guint         prop_id,
                              const GValue *value,
                              GParamSpec   *pspec)
{
  FoundryLicense *self = FOUNDRY_LICENSE (object);

  switch (prop_id)
    {
    case PROP_ID:
      self->id = g_value_dup_string (value);
      break;

    case PROP_TITLE:
      self->title = g_value_dup_string (value);
      break;

    case PROP_TEXT_BYTES:
      self->text = g_value_dup_boxed (value);
      break;

    case PROP_SNIPPET_TEXT_BYTES:
      self->snippet_text = g_value_dup_boxed (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_license_class_init (FoundryLicenseClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_license_finalize;
  object_class->get_property = foundry_license_get_property;
  object_class->set_property = foundry_license_set_property;

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TITLE] =
    g_param_spec_string ("title", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TEXT] =
    g_param_spec_string ("text", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TEXT_BYTES] =
    g_param_spec_boxed ("text-bytes", NULL, NULL,
                        G_TYPE_BYTES,
                        (G_PARAM_READWRITE |
                         G_PARAM_CONSTRUCT_ONLY |
                         G_PARAM_STATIC_STRINGS));

  properties[PROP_SNIPPET_TEXT] =
    g_param_spec_string ("snippet-text", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_SNIPPET_TEXT_BYTES] =
    g_param_spec_boxed ("snippet-text-bytes", NULL, NULL,
                        G_TYPE_BYTES,
                        (G_PARAM_READWRITE |
                         G_PARAM_CONSTRUCT_ONLY |
                         G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_license_init (FoundryLicense *self)
{
}

char *
foundry_license_dup_id (FoundryLicense *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LICENSE (self), NULL);

  return g_strdup (self->id);
}

char *
foundry_license_dup_title (FoundryLicense *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LICENSE (self), NULL);

  return g_strdup (self->title);
}

/**
 * foundry_license_dup_text:
 * @self: a [class@Foundry.License]
 *
 * Returns: (transfer full) (nullable):
 */
GBytes *
foundry_license_dup_text (FoundryLicense *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LICENSE (self), NULL);

  return self->text ? g_bytes_ref (self->text) : NULL;
}

/**
 * foundry_license_dup_snippet_text:
 * @self: a [class@Foundry.License]
 *
 * Returns: (transfer full) (nullable):
 */
GBytes *
foundry_license_dup_snippet_text (FoundryLicense *self)
{
  g_return_val_if_fail (FOUNDRY_IS_LICENSE (self), NULL);

  return self->snippet_text ? g_bytes_ref (self->snippet_text) : NULL;
}

static const struct {
  const char *path;
  const char *id;
  const char *title;
} licenses[] = {
  { "", "NONE", N_("No License") },
  { "agpl_3", "AGPL-3.0-or-later", "AGPL 3.0 or later" },
  { "apache_2", "Apache-2.0", "Apache 2.0" },
  { "eupl_1_2", "EUPL-1.2", "EUPL 1.2" },
  { "gpl_2", "GPL-2.0-or-later", N_("GPL 2.0 or later") },
  { "gpl_3", "GPL-3.0-or-later", N_("GPL 3.0 or later") },
  { "lgpl_2_1", "LGPL-2.1-or-later", N_("LGPL 2.1 or later") },
  { "lgpl_3", "LGPL-3.0-or-later", N_("LGPL 3.0 or later") },
  { "mit_x11", "MIT", "MIT" },
  { "mpl_2", "MPL-2.0", "MPL 2.0" },
};

/**
 * foundry_license_list_all:
 *
 * Get a [iface@Gio.ListModel] of all [class@Foundry.License].
 *
 * Returns: (transfer full):
 */
GListModel *
foundry_license_list_all (void)
{
  static GListModel *model;

  if (g_once_init_enter (&model))
    {
      GListStore *store = g_list_store_new (FOUNDRY_TYPE_LICENSE);

      for (guint i = 0; i < G_N_ELEMENTS (licenses); i++)
        {
          g_autoptr(FoundryLicense) license = NULL;
          g_autoptr(GBytes) text = NULL;
          g_autoptr(GBytes) snippet_text = NULL;
          g_autofree char *text_path = NULL;
          g_autofree char *snippet_text_path = NULL;
          const char *title;

          text_path = g_strdup_printf ("/app/devsuite/foundry/licenses/%s_full", licenses[i].path);
          snippet_text_path = g_strdup_printf ("/app/devsuite/foundry/licenses/%s_short", licenses[i].path);

          text = g_resources_lookup_data (text_path, 0, NULL);
          snippet_text = g_resources_lookup_data (snippet_text_path, 0, NULL);

          title = licenses[i].title ? licenses[i].title : licenses[i].id;

          license = g_object_new (FOUNDRY_TYPE_LICENSE,
                                  "id", licenses[i].id,
                                  "title", title,
                                  "text-bytes", text,
                                  "snippet-text-bytes", snippet_text,
                                  NULL);

          g_list_store_append (store, license);
        }

      g_once_init_leave (&model, G_LIST_MODEL (g_steal_pointer (&store)));
    }

  return g_object_ref (model);
}

/**
 * foundry_license_find:
 * @id: the SPDX identifier
 *
 * Returns: (transfer full) (nullable):
 */
FoundryLicense *
foundry_license_find (const char *id)
{
  g_autoptr(GListModel) model = NULL;
  guint n_items;

  if (id == NULL)
    return NULL;

  model = foundry_license_list_all ();
  n_items = g_list_model_get_n_items (model);

  for (guint i = 0; i < n_items; i++)
    {
      g_autoptr(FoundryLicense) license = g_list_model_get_item (model, i);
      g_autofree char *license_id = foundry_license_dup_id (license);

      if (g_strcmp0 (id, license_id) == 0)
        return g_steal_pointer (&license);
    }

  return NULL;
}
