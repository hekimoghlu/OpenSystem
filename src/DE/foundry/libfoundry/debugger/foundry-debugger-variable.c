/* foundry-debugger-variable.c
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

#include "foundry-debugger-variable.h"
#include "foundry-util.h"

G_DEFINE_ABSTRACT_TYPE (FoundryDebuggerVariable, foundry_debugger_variable, G_TYPE_OBJECT)

enum {
  PROP_0,
  PROP_NAME,
  PROP_TYPE_NAME,
  PROP_VALUE,
  N_PROPS
};

static GParamSpec *properties[N_PROPS];

static void
foundry_debugger_variable_get_property (GObject    *object,
                                        guint       prop_id,
                                        GValue     *value,
                                        GParamSpec *pspec)
{
  FoundryDebuggerVariable *self = FOUNDRY_DEBUGGER_VARIABLE (object);

  switch (prop_id)
    {
    case PROP_NAME:
      g_value_take_string (value, foundry_debugger_variable_dup_name (self));
      break;

    case PROP_TYPE_NAME:
      g_value_take_string (value, foundry_debugger_variable_dup_type_name (self));
      break;

    case PROP_VALUE:
      g_value_take_string (value, foundry_debugger_variable_dup_value (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_variable_class_init (FoundryDebuggerVariableClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_debugger_variable_get_property;

  properties[PROP_NAME] =
    g_param_spec_string ("name", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_TYPE_NAME] =
    g_param_spec_string ("type-name", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_VALUE] =
    g_param_spec_string ("value", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_debugger_variable_init (FoundryDebuggerVariable *self)
{
}

/**
 * foundry_debugger_variable_dup_name:
 * @self: a [class@Foundry.DebuggerVariable]
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_debugger_variable_dup_name (FoundryDebuggerVariable *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_VARIABLE (self), NULL);

  if (FOUNDRY_DEBUGGER_VARIABLE_GET_CLASS (self)->dup_name)
    return FOUNDRY_DEBUGGER_VARIABLE_GET_CLASS (self)->dup_name (self);

  return NULL;
}

/**
 * foundry_debugger_variable_dup_value:
 * @self: a [class@Foundry.DebuggerVariable]
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_debugger_variable_dup_value (FoundryDebuggerVariable *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_VARIABLE (self), NULL);

  if (FOUNDRY_DEBUGGER_VARIABLE_GET_CLASS (self)->dup_value)
    return FOUNDRY_DEBUGGER_VARIABLE_GET_CLASS (self)->dup_value (self);

  return NULL;
}

/**
 * foundry_debugger_variable_dup_type_name:
 * @self: a [class@Foundry.DebuggerVariable]
 *
 * Returns: (transfer full) (nullable):
 */
char *
foundry_debugger_variable_dup_type_name (FoundryDebuggerVariable *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_VARIABLE (self), NULL);

  if (FOUNDRY_DEBUGGER_VARIABLE_GET_CLASS (self)->dup_type_name)
    return FOUNDRY_DEBUGGER_VARIABLE_GET_CLASS (self)->dup_type_name (self);

  return NULL;
}

/**
 * foundry_debugger_variable_is_structured:
 * @self: a [class@Foundry.DebuggerVariable]
 * @n_children: (out) (nullable): the number of known children
 *
 * If the number of children is known, it will be set to @n_children. Otherwise
 * it should be set to zero.
 *
 * Returns: `True` if @self is known to have children that may be queried;
 *   otherwise `False`.
 *
 * Since: 1.1
 */
gboolean
foundry_debugger_variable_is_structured (FoundryDebuggerVariable *self,
                                         guint                   *n_children)
{
  guint dummy = 0;

  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_VARIABLE (self), FALSE);

  if (n_children == NULL)
    n_children = &dummy;

  *n_children = 0;

  if (FOUNDRY_DEBUGGER_VARIABLE_GET_CLASS (self)->is_structured)
    return FOUNDRY_DEBUGGER_VARIABLE_GET_CLASS (self)->is_structured (self, n_children);

  return FALSE;
}

/**
 * foundry_debugger_variable_list_children:
 * @self: a [class@Foundry.DebuggerVariable]
 *
 * Queries the structured children of the variable.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [iface@Gio.ListModel] of [class@Foundry.DebuggerVariable] or
 *   rejects with error.
 *
 * Since: 1.1
 */
DexFuture *
foundry_debugger_variable_list_children (FoundryDebuggerVariable *self)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER_VARIABLE (self));

  if (FOUNDRY_DEBUGGER_VARIABLE_GET_CLASS (self)->list_children)
    return FOUNDRY_DEBUGGER_VARIABLE_GET_CLASS (self)->list_children (self);

  return foundry_future_new_not_supported ();
}

/**
 * foundry_debugger_variable_read_memory:
 * @self: a [class@Foundry.DebuggerVariable]
 * @offset: offset to begin reading from
 * @count: number of bytes to read, must be > 0
 *
 * Read @count bytes at @offset of the variable.
 *
 * Returns: (transfer full): a [class@Dex.Future] that resolves to a
 *   [struct@GLib.Bytes] or rejects with error.
 *
 * Sicne: 1.1
 */
DexFuture *
foundry_debugger_variable_read_memory (FoundryDebuggerVariable *self,
                                       guint64                  offset,
                                       guint64                  count)
{
  dex_return_error_if_fail (FOUNDRY_IS_DEBUGGER_VARIABLE (self));
  dex_return_error_if_fail (count > 0);

  if (FOUNDRY_DEBUGGER_VARIABLE_GET_CLASS (self)->read_memory)
    return FOUNDRY_DEBUGGER_VARIABLE_GET_CLASS (self)->read_memory (self, offset, count);

  return foundry_future_new_not_supported ();
}
