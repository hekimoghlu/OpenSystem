/* foundry-flatpak-manifest-loader.c
 *
 * Copyright 2015 Red Hat, Inc
 * Copyright 2023 GNOME Foundation Inc.
 * Copyright 2025 Christian Hergert
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

#include <foundry.h>
#include <yaml.h>

#include "foundry-flatpak-list.h"
#include "foundry-flatpak-manifest.h"
#include "foundry-flatpak-manifest-loader-private.h"
#include "foundry-flatpak-serializable-private.h"

struct _FoundryFlatpakManifestLoader
{
  GObject  parent_instance;
  GFile   *file;
  GFile   *base_dir;
};

enum {
  PROP_0,
  PROP_FILE,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryFlatpakManifestLoader, foundry_flatpak_manifest_loader, G_TYPE_OBJECT)

G_DEFINE_AUTO_CLEANUP_CLEAR_FUNC (yaml_parser_t, yaml_parser_delete)
G_DEFINE_AUTO_CLEANUP_CLEAR_FUNC (yaml_document_t, yaml_document_delete)

static GParamSpec *properties[N_PROPS];

static void
foundry_flatpak_manifest_loader_finalize (GObject *object)
{
  FoundryFlatpakManifestLoader *self = (FoundryFlatpakManifestLoader *)object;

  g_clear_object (&self->file);
  g_clear_object (&self->base_dir);

  G_OBJECT_CLASS (foundry_flatpak_manifest_loader_parent_class)->finalize (object);
}

static void
foundry_flatpak_manifest_loader_get_property (GObject    *object,
                                             guint       prop_id,
                                             GValue     *value,
                                             GParamSpec *pspec)
{
  FoundryFlatpakManifestLoader *self = FOUNDRY_FLATPAK_MANIFEST_LOADER (object);

  switch (prop_id)
    {
    case PROP_FILE:
      g_value_set_object (value, self->file);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_manifest_loader_set_property (GObject      *object,
                                             guint         prop_id,
                                             const GValue *value,
                                             GParamSpec   *pspec)
{
  FoundryFlatpakManifestLoader *self = FOUNDRY_FLATPAK_MANIFEST_LOADER (object);

  switch (prop_id)
    {
    case PROP_FILE:
      if ((self->file = g_value_dup_object (value)))
        self->base_dir = g_file_get_parent (self->file);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_manifest_loader_class_init (FoundryFlatpakManifestLoaderClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_flatpak_manifest_loader_finalize;
  object_class->get_property = foundry_flatpak_manifest_loader_get_property;
  object_class->set_property = foundry_flatpak_manifest_loader_set_property;

  properties[PROP_FILE] =
    g_param_spec_object ("file", NULL, NULL,
                         G_TYPE_FILE,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_flatpak_manifest_loader_init (FoundryFlatpakManifestLoader *self)
{
}

/**
 * foundry_flatpak_manifest_loader_new:
 * @file: a [iface@Gio.File]
 *
 * Returns: (transfer full):
 */
FoundryFlatpakManifestLoader *
foundry_flatpak_manifest_loader_new (GFile *file)
{
  g_return_val_if_fail (G_IS_FILE (file), NULL);

  return g_object_new (FOUNDRY_TYPE_FLATPAK_MANIFEST_LOADER,
                       "file", file,
                       NULL);
}

/**
 * foundry_flatpak_manifest_loader_dup_file:
 * @self: a [class@Foundry.FlatpakManifestLoader]
 *
 * Returns: (transfer full):
 */
GFile *
foundry_flatpak_manifest_loader_dup_file (FoundryFlatpakManifestLoader *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MANIFEST_LOADER (self), NULL);

  return g_object_ref (self->file);
}

/**
 * foundry_flatpak_manifest_loader_dup_base_dir:
 * @self: a [class@Foundry.FlatpakManifestLoader]
 *
 * Returns: (transfer full):
 */
GFile *
foundry_flatpak_manifest_loader_dup_base_dir (FoundryFlatpakManifestLoader *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MANIFEST_LOADER (self), NULL);

  return g_object_ref (self->base_dir);
}

static JsonNode *
_yaml_node_to_json (yaml_document_t *doc,
                    yaml_node_t     *node)
{
  g_autoptr(JsonObject) object = NULL;
  g_autoptr(JsonArray) array = NULL;
  const char *scalar = NULL;
  yaml_node_item_t *item = NULL;
  yaml_node_pair_t *pair = NULL;
  JsonNode *json;

  g_assert (doc != NULL);
  g_assert (node != NULL);

  json = json_node_alloc ();

  switch (node->type)
    {
    case YAML_NO_NODE:
      json_node_init_null (json);
      break;

    case YAML_SCALAR_NODE:
      scalar = (char *) node->data.scalar.value;

      if (node->data.scalar.style == YAML_PLAIN_SCALAR_STYLE)
        {
          if (strcmp (scalar, "true") == 0)
            {
              json_node_init_boolean (json, TRUE);
              break;
            }
          else if (strcmp (scalar, "false") == 0)
            {
              json_node_init_boolean (json, FALSE);
              break;
            }
          else if (strcmp (scalar, "null") == 0)
            {
              json_node_init_null (json);
              break;
            }

          if (*scalar != '\0')
            {
              char *endptr;
              gint64 num = g_ascii_strtoll (scalar, &endptr, 10);

              if (*endptr == '\0')
                {
                  json_node_init_int (json, num);
                  break;
                }
              else if (*endptr == '.' && (endptr != scalar || endptr[1] != '\0'))
                {
                  /* Make sure that N.N, N., and .N (where N is a digit) are picked up as numbers. */
                  g_ascii_strtoll (endptr + 1, &endptr, 10);
                }
            }
        }

      json_node_init_string (json, scalar);
      break;

    case YAML_SEQUENCE_NODE:
      array = json_array_new ();

      for (item = node->data.sequence.items.start; item < node->data.sequence.items.top; item++)
        {
          yaml_node_t *child = yaml_document_get_node (doc, *item);
          if (child != NULL)
            json_array_add_element (array, _yaml_node_to_json (doc, child));
        }

      json_node_init_array (json, array);
      break;

    case YAML_MAPPING_NODE:
      object = json_object_new ();

      for (pair = node->data.mapping.pairs.start; pair < node->data.mapping.pairs.top; pair++)
        {
          yaml_node_t *key = yaml_document_get_node (doc, pair->key);
          yaml_node_t *value = yaml_document_get_node (doc, pair->value);

          g_warn_if_fail (key->type == YAML_SCALAR_NODE);
          json_object_set_member (object, (char *) key->data.scalar.value,
                                  _yaml_node_to_json (doc, value));
        }

      json_node_init_object (json, object);
      break;

    default:
      break;
    }

  return json;
}

static JsonNode *
parse_yaml_to_json (GBytes  *contents,
                    GError **error)
{
  g_auto(yaml_parser_t) parser = {0};
  g_auto(yaml_document_t) doc = {{0}};
  const yaml_char_t *data;
  yaml_node_t *root;
  gsize size;

  if (!yaml_parser_initialize (&parser))
    {
      g_set_error_literal (error,
                           G_IO_ERROR,
                           G_IO_ERROR_FAILED,
                           "Failed to initialize Yaml parser");
      return NULL;
    }

  data = g_bytes_get_data (contents, &size);

  yaml_parser_set_input_string (&parser, data, size);

  if (!yaml_parser_load (&parser, &doc))
    {
      g_set_error (error,
                   G_IO_ERROR,
                   G_IO_ERROR_FAILED,
                   "%zu:%zu: %s",
                   parser.problem_mark.line + 1,
                   parser.problem_mark.column + 1,
                   parser.problem);
      return NULL;
    }

  if (!(root = yaml_document_get_root_node (&doc)))
    {
      g_set_error_literal (error,
                           G_IO_ERROR,
                           G_IO_ERROR_FAILED,
                           "Document has no root node.");
      return NULL;
    }

  return _yaml_node_to_json (&doc, root);
}

DexFuture *
_foundry_flatpak_manifest_load_file_as_json (GFile *file)
{
  g_autoptr(JsonNode) root = NULL;
  g_autoptr(GError) error = NULL;
  g_autofree char *basename = NULL;

  g_return_val_if_fail (G_IS_FILE (file), NULL);

  basename = g_file_get_basename (file);

  if (g_str_has_suffix (basename, ".yaml") ||
      g_str_has_suffix (basename, ".yml"))
    {
      g_autoptr(GBytes) bytes = NULL;

      if (!(bytes = dex_await_boxed (dex_file_load_contents_bytes (file), &error)))
        return dex_future_new_for_error (g_steal_pointer (&error));

      if (!(root = parse_yaml_to_json (bytes, &error)))
        return dex_future_new_for_error (g_steal_pointer (&error));
    }
  else
    {
      g_autoptr(JsonParser) parser = json_parser_new_immutable ();

      if (!dex_await (foundry_json_parser_load_from_file (parser, file), &error))
        return dex_future_new_for_error (g_steal_pointer (&error));

      root = json_node_ref (json_parser_get_root (parser));
    }

  return dex_future_new_take_boxed (JSON_TYPE_NODE, g_steal_pointer (&root));
}

static DexFuture *
foundry_flatpak_manifest_loader_load_fiber (gpointer data)
{
  FoundryFlatpakManifestLoader *self = data;
  g_autoptr(FoundryFlatpakManifest) manifest = NULL;
  g_autoptr(JsonNode) root = NULL;
  g_autoptr(GError) error = NULL;

  g_assert (FOUNDRY_IS_FLATPAK_MANIFEST_LOADER (self));

  if (!(root = dex_await_boxed (_foundry_flatpak_manifest_load_file_as_json (self->file), &error)))
    return dex_future_new_for_error (g_steal_pointer (&error));

  if (!(manifest = dex_await_object (_foundry_flatpak_manifest_loader_deserialize (self, FOUNDRY_TYPE_FLATPAK_MANIFEST, root), &error)))
      return dex_future_new_for_error (g_steal_pointer (&error));

  return dex_future_new_take_object (g_steal_pointer (&manifest));
}

/**
 * foundry_flatpak_manifest_loader_load:
 * @self: a [class@Foundry.FlatpakManifestLoader]
 *
 * Returns: (transfer full):
 */
DexFuture *
foundry_flatpak_manifest_loader_load (FoundryFlatpakManifestLoader *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_FLATPAK_MANIFEST_LOADER (self));

  return dex_scheduler_spawn (NULL, 0,
                              foundry_flatpak_manifest_loader_load_fiber,
                              g_object_ref (self),
                              g_object_unref);
}

DexFuture *
_foundry_flatpak_manifest_loader_deserialize (FoundryFlatpakManifestLoader *self,
                                              GType                         type,
                                              JsonNode                     *node)
{
  g_autoptr(GObject) object = NULL;

  g_return_val_if_fail (FOUNDRY_IS_FLATPAK_MANIFEST_LOADER (self), NULL);
  g_return_val_if_fail (g_type_is_a (type, G_TYPE_OBJECT), NULL);
  g_return_val_if_fail (node != NULL, NULL);

  if (g_type_is_a (type, FOUNDRY_TYPE_FLATPAK_SERIALIZABLE))
    {
      g_autoptr(FoundryFlatpakSerializable) serializable = NULL;

      serializable = _foundry_flatpak_serializable_new (type, self->base_dir);
      return _foundry_flatpak_serializable_deserialize (serializable, node);
    }

  if (JSON_NODE_HOLDS_NULL (node))
    return dex_future_new_take_object (NULL);

  if ((object = json_gobject_deserialize (type, node)))
    return dex_future_new_take_object (g_steal_pointer (&object));

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_FAILED,
                                "Failed to deserialize type \"%s\"",
                                g_type_name (type));
}
