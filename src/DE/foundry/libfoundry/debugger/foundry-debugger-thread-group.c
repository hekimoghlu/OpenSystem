/* foundry-debugger-thread-group.c
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

#include "foundry-debugger-thread-group.h"

enum {
  PROP_0,
  PROP_ID,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryDebuggerThreadGroup, foundry_debugger_thread_group, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_debugger_thread_group_get_property (GObject    *object,
                                            guint       prop_id,
                                            GValue     *value,
                                            GParamSpec *pspec)
{
  FoundryDebuggerThreadGroup *self = FOUNDRY_DEBUGGER_THREAD_GROUP (object);

  switch (prop_id)
    {
    case PROP_ID:
      g_value_take_string (value, foundry_debugger_thread_group_dup_id (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_thread_group_class_init (FoundryDebuggerThreadGroupClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_debugger_thread_group_get_property;

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_debugger_thread_group_init (FoundryDebuggerThreadGroup *self)
{
}

char *
foundry_debugger_thread_group_dup_id (FoundryDebuggerThreadGroup *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_THREAD_GROUP (self), NULL);

  if (FOUNDRY_DEBUGGER_THREAD_GROUP_GET_CLASS (self)->dup_id)
    return FOUNDRY_DEBUGGER_THREAD_GROUP_GET_CLASS (self)->dup_id (self);

  return NULL;
}
