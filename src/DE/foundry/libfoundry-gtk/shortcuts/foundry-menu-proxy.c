/* foundry-menu-proxy.c
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

#include "foundry-menu-manager.h"
#include "foundry-menu-proxy.h"

struct _FoundryMenuProxy
{
  GMenuModel parent_instance;
  GMenuModel *menu_model;
  char *menu_id;
};

enum {
  PROP_0,
  PROP_MENU_ID,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryMenuProxy, foundry_menu_proxy, G_TYPE_MENU_MODEL)

static GParamSpec *properties[N_PROPS];

static gboolean
foundry_menu_proxy_is_mutable (GMenuModel *menu_model)
{
  return TRUE;
}

static int
foundry_menu_proxy_get_n_items (GMenuModel *menu_model)
{
  FoundryMenuProxy *self = FOUNDRY_MENU_PROXY (menu_model);

  return self->menu_model ? g_menu_model_get_n_items (self->menu_model) : 0;
}

static GMenuAttributeIter *
foundry_menu_proxy_iterate_item_attributes (GMenuModel *menu_model,
                                            int         item_index)
{
  FoundryMenuProxy *self = FOUNDRY_MENU_PROXY (menu_model);

  if (self->menu_model)
    return g_menu_model_iterate_item_attributes (self->menu_model, item_index);

  return NULL;
}

static GMenuLinkIter *
foundry_menu_proxy_iterate_item_links (GMenuModel *menu_model,
                                       int         item_index)
{
  FoundryMenuProxy *self = FOUNDRY_MENU_PROXY (menu_model);

  if (self->menu_model)
    return g_menu_model_iterate_item_links (self->menu_model, item_index);

  return NULL;
}

static GMenuModel *
foundry_menu_proxy_get_item_link (GMenuModel *menu_model,
                                  int         item_index,
                                  const char *link)
{
  FoundryMenuProxy *self = FOUNDRY_MENU_PROXY (menu_model);

  if (self->menu_model)
    return g_menu_model_get_item_link (self->menu_model, item_index, link);

  return NULL;
}

static void
foundry_menu_proxy_dispose (GObject *object)
{
  FoundryMenuProxy *self = (FoundryMenuProxy *)object;

  foundry_menu_proxy_set_menu_id (self, NULL);

  g_clear_object (&self->menu_model);
  g_clear_pointer (&self->menu_id, g_free);

  G_OBJECT_CLASS (foundry_menu_proxy_parent_class)->dispose (object);
}

static void
foundry_menu_proxy_get_property (GObject    *object,
                                 guint       prop_id,
                                 GValue     *value,
                                 GParamSpec *pspec)
{
  FoundryMenuProxy *self = FOUNDRY_MENU_PROXY (object);

  switch (prop_id)
    {
    case PROP_MENU_ID:
      g_value_set_string (value, foundry_menu_proxy_get_menu_id (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_menu_proxy_set_property (GObject      *object,
                                 guint         prop_id,
                                 const GValue *value,
                                 GParamSpec   *pspec)
{
  FoundryMenuProxy *self = FOUNDRY_MENU_PROXY (object);

  switch (prop_id)
    {
    case PROP_MENU_ID:
      foundry_menu_proxy_set_menu_id (self, g_value_get_string (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_menu_proxy_class_init (FoundryMenuProxyClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  GMenuModelClass *menu_model_class = G_MENU_MODEL_CLASS (klass);

  object_class->dispose = foundry_menu_proxy_dispose;
  object_class->get_property = foundry_menu_proxy_get_property;
  object_class->set_property = foundry_menu_proxy_set_property;

  menu_model_class->is_mutable = foundry_menu_proxy_is_mutable;
  menu_model_class->get_n_items = foundry_menu_proxy_get_n_items;
  menu_model_class->iterate_item_attributes = foundry_menu_proxy_iterate_item_attributes;
  menu_model_class->iterate_item_links = foundry_menu_proxy_iterate_item_links;
  menu_model_class->get_item_link = foundry_menu_proxy_get_item_link;

  properties[PROP_MENU_ID] =
    g_param_spec_string ("menu-id", NULL, NULL,
                         NULL,
                         (G_PARAM_READWRITE |
                          G_PARAM_EXPLICIT_NOTIFY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_menu_proxy_init (FoundryMenuProxy *self)
{
}

FoundryMenuProxy *
foundry_menu_proxy_new (const char *menu_id)
{
  return g_object_new (FOUNDRY_TYPE_MENU_PROXY,
                       "menu-id", menu_id,
                       NULL);
}

const char *
foundry_menu_proxy_get_menu_id (FoundryMenuProxy *self)
{
  g_return_val_if_fail (FOUNDRY_IS_MENU_PROXY (self), NULL);

  return self->menu_id;
}

void
foundry_menu_proxy_set_menu_id (FoundryMenuProxy *self,
                                const char       *menu_id)
{
  guint old_len;

  g_return_if_fail (FOUNDRY_IS_MENU_PROXY (self));

  old_len = g_menu_model_get_n_items (G_MENU_MODEL (self));

  if (g_set_str (&self->menu_id, menu_id))
    {
      FoundryMenuManager *menu_manager = foundry_menu_manager_get_default ();
      GMenuModel *menu_model = NULL;

      if (menu_id != NULL)
        menu_model = G_MENU_MODEL (foundry_menu_manager_get_menu_by_id (menu_manager, menu_id));

      if (self->menu_model != menu_model)
        {
          guint new_len;

          if (self->menu_model)
            {
              g_signal_handlers_disconnect_by_func (self->menu_model,
                                                    G_CALLBACK (g_menu_model_items_changed),
                                                    self);
              g_clear_object (&self->menu_model);
            }

          if (menu_model)
            {
              self->menu_model = g_object_ref (menu_model);
              g_signal_connect_object (self->menu_model,
                                       "items-changed",
                                       G_CALLBACK (g_menu_model_items_changed),
                                       self,
                                       G_CONNECT_SWAPPED);
            }

          new_len = g_menu_model_get_n_items (G_MENU_MODEL (self));
          g_menu_model_items_changed (G_MENU_MODEL (self), 0, old_len, new_len);
        }

      g_object_notify_by_pspec (G_OBJECT (self), properties[PROP_MENU_ID]);
    }
}
