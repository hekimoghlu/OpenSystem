/* foundry-flatpak-list.c
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

#include "foundry-flatpak-list-private.h"
#include "foundry-flatpak-manifest-loader-private.h"
#include "foundry-flatpak-serializable-private.h"

typedef struct
{
  GPtrArray *items;
} FoundryFlatpakListPrivate;

static void list_model_iface_init (GListModelInterface *iface);

G_DEFINE_ABSTRACT_TYPE_WITH_CODE (FoundryFlatpakList, foundry_flatpak_list, FOUNDRY_TYPE_FLATPAK_SERIALIZABLE,
                                  G_ADD_PRIVATE (FoundryFlatpakList)
                                  G_IMPLEMENT_INTERFACE (G_TYPE_LIST_MODEL, list_model_iface_init))

enum {
  PROP_0,
  PROP_ITEM_TYPE,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static GType
find_item_type (FoundryFlatpakList *self,
                JsonNode           *node)
{
  if (JSON_NODE_HOLDS_OBJECT (node))
    {
      JsonObject *object = json_node_get_object (node);

      if (json_object_has_member (object, "type"))
        {
          const char *type = json_object_get_string_member (object, "type");

          if (type != NULL)
            return FOUNDRY_FLATPAK_LIST_GET_CLASS (self)->get_item_type (self, type);
        }
    }

  return FOUNDRY_FLATPAK_LIST_GET_CLASS (self)->item_type;
}

static DexFuture *
foundry_flatpak_list_deserialize (FoundryFlatpakSerializable *serializable,
                                  JsonNode                   *node)
{
  FoundryFlatpakList *self = FOUNDRY_FLATPAK_LIST (serializable);
  g_autoptr(GFile) base_dir = _foundry_flatpak_serializable_dup_base_dir (serializable);

  if (0) { }
  else if (JSON_NODE_HOLDS_ARRAY (node))
    {
      JsonArray *ar = json_node_get_array (node);
      gsize len = json_array_get_length (ar);

      /* In this mode, we have a simple [{..}, {..}] style array of
       * objects for the list.
       */

      len = json_array_get_length (ar);

      for (gsize i = 0; i < len; i++)
        {
          JsonNode *element = json_array_get_element (ar, i);
          g_autoptr(FoundryFlatpakSerializable) child = NULL;
          g_autoptr(JsonNode) alloc_node = NULL;
          g_autoptr(GFile) element_base_dir = NULL;
          g_autoptr(GError) error = NULL;
          GType child_item_type;

          /* An oddity that is sometimes used is a string filename here
           * that links to an object array in another file. That is really
           * meant to extend this array rather than be a sub-object.
           */

          g_set_object (&element_base_dir, base_dir);

          if (JSON_NODE_HOLDS_VALUE (element) &&
              json_node_get_value_type (element) == G_TYPE_STRING)
            {
              const char *subpath = json_node_get_string (element);
              g_autoptr(JsonNode) subnode = NULL;
              g_autoptr(GFile) subfile = NULL;
              g_autoptr(GFile) sub_base_dir = NULL;

              if (!(subfile = foundry_flatpak_serializable_resolve_file (FOUNDRY_FLATPAK_SERIALIZABLE (self), subpath, &error)))
                return dex_future_new_for_error (g_steal_pointer (&error));

              g_clear_object (&element_base_dir);
              element_base_dir = g_file_get_parent (subfile);

              if (!(subnode = dex_await_boxed (_foundry_flatpak_manifest_load_file_as_json (subfile), &error)))
                return dex_future_new_for_error (g_steal_pointer (&error));

              if (JSON_NODE_HOLDS_ARRAY (subnode))
                {
                  JsonArray *subarray = json_node_get_array (subnode);
                  gsize sublen = json_array_get_length (subarray);

                  for (guint j = 0; j < sublen; j++)
                    {
                      JsonNode *subelement = json_array_get_element (subarray, j);
                      g_autoptr(FoundryFlatpakSerializable) subchild = NULL;
                      GType sub_item_type = find_item_type (self, subelement);

                      subchild = _foundry_flatpak_serializable_new (sub_item_type, element_base_dir);

                      if (!dex_await (_foundry_flatpak_serializable_deserialize (subchild, subelement), &error))
                        return dex_future_new_for_error (g_steal_pointer (&error));

                      foundry_flatpak_list_add (FOUNDRY_FLATPAK_LIST (serializable), subchild);
                    }
                }
              else if (JSON_NODE_HOLDS_OBJECT (subnode))
                {
                  element = alloc_node = json_node_ref (subnode);
                  goto handle_child_object;
                }
              else
                {
                  return dex_future_new_reject (G_IO_ERROR,
                                                G_IO_ERROR_INVALID_DATA,
                                                "Unspected root node type in \"%s\"",
                                                subpath);
                }
            }
          else
            {
            handle_child_object:
              child_item_type = find_item_type (self, element);

              if (G_TYPE_IS_ABSTRACT (child_item_type))
                return dex_future_new_reject (G_IO_ERROR,
                                              G_IO_ERROR_INVALID_DATA,
                                              "Unknown type defined in manifest");

              child = _foundry_flatpak_serializable_new (child_item_type, element_base_dir);

              if (!dex_await (_foundry_flatpak_serializable_deserialize (child, element), &error))
                return dex_future_new_for_error (g_steal_pointer (&error));

              foundry_flatpak_list_add (FOUNDRY_FLATPAK_LIST (serializable), child);
            }
        }
    }
  else if (JSON_NODE_HOLDS_OBJECT (node))
    {
      JsonObject *object = json_node_get_object (node);
      JsonObjectIter iter;
      const char *member_name = NULL;
      JsonNode *member_node = NULL;
      GType child_item_type;

      /* In this mode, we have a list that is keyed by the name of
       * the child item type.
       *
       * For example:
       *
       * "add-extensions" : { "name" : { ... } }
       */

      json_object_iter_init (&iter, object);
      while (json_object_iter_next (&iter, &member_name, &member_node))
        {
          g_autoptr(FoundryFlatpakSerializable) child = NULL;
          g_autoptr(GError) error = NULL;
          g_auto(GValue) value = G_VALUE_INIT;
          GParamSpec *pspec;

          child_item_type = find_item_type (self, member_node);
          child = _foundry_flatpak_serializable_new (child_item_type, base_dir);

          if (!dex_await (_foundry_flatpak_serializable_deserialize (child, member_node), &error))
            return dex_future_new_for_error (g_steal_pointer (&error));

          if (!(pspec = g_object_class_find_property (G_OBJECT_GET_CLASS (child), "name")) ||
              pspec->value_type != G_TYPE_STRING)
            return dex_future_new_reject (G_IO_ERROR,
                                          G_IO_ERROR_FAILED,
                                          "Object \"%s\" msising name property",
                                          G_OBJECT_TYPE_NAME (child));

          g_value_init (&value, G_TYPE_STRING);
          g_value_set_string (&value, member_name);
          g_object_set_property (G_OBJECT (child), "name", &value);

          foundry_flatpak_list_add (FOUNDRY_FLATPAK_LIST (serializable), child);
        }
    }

  return dex_future_new_true ();
}

static GType
foundry_flatpak_list_real_get_item_type (FoundryFlatpakList *self,
                                         const char         *type)
{
  return FOUNDRY_FLATPAK_LIST_GET_CLASS (self)->item_type;
}

static void
foundry_flatpak_list_constructed (GObject *object)
{
  FoundryFlatpakList *self = (FoundryFlatpakList *)object;

  G_OBJECT_CLASS (foundry_flatpak_list_parent_class)->constructed (object);

  if (!g_type_is_a (FOUNDRY_FLATPAK_LIST_GET_CLASS (self)->item_type, FOUNDRY_TYPE_FLATPAK_SERIALIZABLE))
    g_error ("Attempt to create %s without a serializable item-type of \"%s\"\n",
             G_OBJECT_TYPE_NAME (self),
             g_type_name (FOUNDRY_FLATPAK_LIST_GET_CLASS (self)->item_type));
}

static void
foundry_flatpak_list_dispose (GObject *object)
{
  FoundryFlatpakList *self = (FoundryFlatpakList *)object;
  FoundryFlatpakListPrivate *priv = foundry_flatpak_list_get_instance_private (self);

  if (priv->items->len > 0)
    g_ptr_array_remove_range (priv->items, 0, priv->items->len);

  G_OBJECT_CLASS (foundry_flatpak_list_parent_class)->dispose (object);
}

static void
foundry_flatpak_list_finalize (GObject *object)
{
  FoundryFlatpakList *self = (FoundryFlatpakList *)object;
  FoundryFlatpakListPrivate *priv = foundry_flatpak_list_get_instance_private (self);

  g_clear_pointer (&priv->items, g_ptr_array_unref);

  G_OBJECT_CLASS (foundry_flatpak_list_parent_class)->finalize (object);
}

static void
foundry_flatpak_list_get_property (GObject    *object,
                                   guint       prop_id,
                                   GValue     *value,
                                   GParamSpec *pspec)
{
  switch (prop_id)
    {
    case PROP_ITEM_TYPE:
      g_value_set_gtype (value, FOUNDRY_FLATPAK_LIST_GET_CLASS (object)->item_type);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_flatpak_list_class_init (FoundryFlatpakListClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryFlatpakSerializableClass *serializable_class = FOUNDRY_FLATPAK_SERIALIZABLE_CLASS (klass);

  object_class->constructed = foundry_flatpak_list_constructed;
  object_class->dispose = foundry_flatpak_list_dispose;
  object_class->finalize = foundry_flatpak_list_finalize;
  object_class->get_property = foundry_flatpak_list_get_property;

  serializable_class->deserialize = foundry_flatpak_list_deserialize;

  klass->get_item_type = foundry_flatpak_list_real_get_item_type;

  properties[PROP_ITEM_TYPE] =
    g_param_spec_gtype ("item-type", NULL, NULL,
                        G_TYPE_OBJECT,
                        (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_flatpak_list_init (FoundryFlatpakList *self)
{
  FoundryFlatpakListPrivate *priv = foundry_flatpak_list_get_instance_private (self);

  priv->items = g_ptr_array_new_with_free_func (g_object_unref);
}

void
foundry_flatpak_list_add (FoundryFlatpakList *self,
                          gpointer            instance)
{
  FoundryFlatpakListPrivate *priv = foundry_flatpak_list_get_instance_private (self);

  g_return_if_fail (FOUNDRY_IS_FLATPAK_LIST (self));
  g_return_if_fail (G_TYPE_CHECK_INSTANCE_TYPE (instance, FOUNDRY_FLATPAK_LIST_GET_CLASS (self)->item_type));

  g_ptr_array_add (priv->items, g_object_ref (instance));

  g_list_model_items_changed (G_LIST_MODEL (self), priv->items->len - 1, 0, 1);
}

static guint
foundry_flatpak_list_get_n_items (GListModel *model)
{
  FoundryFlatpakList *self = FOUNDRY_FLATPAK_LIST (model);
  FoundryFlatpakListPrivate *priv = foundry_flatpak_list_get_instance_private (self);

  return priv->items->len;
}

static gpointer
foundry_flatpak_list_get_item (GListModel *model,
                               guint       position)
{
  FoundryFlatpakList *self = FOUNDRY_FLATPAK_LIST (model);
  FoundryFlatpakListPrivate *priv = foundry_flatpak_list_get_instance_private (self);

  if (position >= priv->items->len)
    return NULL;

  return g_object_ref (g_ptr_array_index (priv->items, position));
}

static GType
foundry_flatpak_list_get_item_type (GListModel *model)
{
  return FOUNDRY_FLATPAK_LIST_GET_CLASS (model)->item_type;
}

static void
list_model_iface_init (GListModelInterface *iface)
{
  iface->get_n_items = foundry_flatpak_list_get_n_items;
  iface->get_item = foundry_flatpak_list_get_item;
  iface->get_item_type = foundry_flatpak_list_get_item_type;
}
