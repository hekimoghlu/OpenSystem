/* foundry-debugger-module.c
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

#include "foundry-debugger-mapped-region.h"
#include "foundry-debugger-module.h"

enum {
  PROP_0,
  PROP_ADDRESS_SPACE,
  PROP_HOST_PATH,
  PROP_ID,
  PROP_NAME,
  PROP_PATH,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryDebuggerModule, foundry_debugger_module, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_debugger_module_get_property (GObject    *object,
                                      guint       prop_id,
                                      GValue     *value,
                                      GParamSpec *pspec)
{
  FoundryDebuggerModule *self = FOUNDRY_DEBUGGER_MODULE (object);

  switch (prop_id)
    {
    case PROP_ADDRESS_SPACE:
      g_value_take_object (value, foundry_debugger_module_list_address_space (self));
      break;

    case PROP_HOST_PATH:
      g_value_take_string (value, foundry_debugger_module_dup_host_path (self));
      break;

    case PROP_ID:
      g_value_take_string (value, foundry_debugger_module_dup_id (self));
      break;

    case PROP_NAME:
      g_value_take_string (value, foundry_debugger_module_dup_name (self));
      break;

    case PROP_PATH:
      g_value_take_string (value, foundry_debugger_module_dup_path (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_module_class_init (FoundryDebuggerModuleClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_debugger_module_get_property;

  properties[PROP_ADDRESS_SPACE] =
    g_param_spec_object ("address-space", NULL, NULL,
                         G_TYPE_LIST_MODEL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_HOST_PATH] =
    g_param_spec_string ("host-path", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  /**
   * FoundryDebuggerModule:name:
   *
   * Since: 1.1
   */
  properties[PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_PATH] =
    g_param_spec_string ("path", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_debugger_module_init (FoundryDebuggerModule *self)
{
}

char *
foundry_debugger_module_dup_id (FoundryDebuggerModule *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_MODULE (self), NULL);

  if (FOUNDRY_DEBUGGER_MODULE_GET_CLASS (self)->dup_id)
    return FOUNDRY_DEBUGGER_MODULE_GET_CLASS (self)->dup_id (self);

  return NULL;
}

/**
 * foundry_debugger_module_dup_name:
 * @self: a [class@Foundry.DebuggerModule]
 *
 * Returns: (transfer full):
 *
 * Since: 1.1
 */
char *
foundry_debugger_module_dup_name (FoundryDebuggerModule *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_MODULE (self), NULL);

  if (FOUNDRY_DEBUGGER_MODULE_GET_CLASS (self)->dup_name)
    return FOUNDRY_DEBUGGER_MODULE_GET_CLASS (self)->dup_name (self);

  return foundry_debugger_module_dup_id (self);
}

char *
foundry_debugger_module_dup_path (FoundryDebuggerModule *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_MODULE (self), NULL);

  if (FOUNDRY_DEBUGGER_MODULE_GET_CLASS (self)->dup_path)
    return FOUNDRY_DEBUGGER_MODULE_GET_CLASS (self)->dup_path (self);

  return NULL;
}

char *
foundry_debugger_module_dup_host_path (FoundryDebuggerModule *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_MODULE (self), NULL);

  if (FOUNDRY_DEBUGGER_MODULE_GET_CLASS (self)->dup_host_path)
    return FOUNDRY_DEBUGGER_MODULE_GET_CLASS (self)->dup_host_path (self);

  return NULL;
}

/**
 * foundry_debugger_module_list_address_space:
 * @self: a [class@Foundry.DebuggerModule]
 *
 * Returns: (transfer full): a [iface@Gio.ListModel] of
 *   [class@Foundry.DebuggerMappedRegion]
 */
GListModel *
foundry_debugger_module_list_address_space (FoundryDebuggerModule *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_MODULE (self), NULL);

  if (FOUNDRY_DEBUGGER_MODULE_GET_CLASS (self)->list_address_space)
    return FOUNDRY_DEBUGGER_MODULE_GET_CLASS (self)->list_address_space (self);

  return G_LIST_MODEL (g_list_store_new (FOUNDRY_TYPE_DEBUGGER_MAPPED_REGION));
}
