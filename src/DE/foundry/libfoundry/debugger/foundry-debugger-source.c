/* foundry-debugger-source.c
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

#include "foundry-debugger-source.h"

enum {
  PROP_0,
  PROP_ID,
  PROP_NAME,
  PROP_PATH,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryDebuggerSource, foundry_debugger_source, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_debugger_source_get_property (GObject    *object,
                                      guint       prop_id,
                                      GValue     *value,
                                      GParamSpec *pspec)
{
  FoundryDebuggerSource *self = FOUNDRY_DEBUGGER_SOURCE (object);

  switch (prop_id)
    {
    case PROP_ID:
      g_value_take_string (value, foundry_debugger_source_dup_id (self));
      break;

    case PROP_NAME:
      g_value_take_string (value, foundry_debugger_source_dup_name (self));
      break;

    case PROP_PATH:
      g_value_take_string (value, foundry_debugger_source_dup_path (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_source_class_init (FoundryDebuggerSourceClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_debugger_source_get_property;

  properties[PROP_ID] =
    g_param_spec_string ("id", NULL, NULL,
                         NULL,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

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
foundry_debugger_source_init (FoundryDebuggerSource *self)
{
}

char *
foundry_debugger_source_dup_id (FoundryDebuggerSource *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_SOURCE (self), NULL);

  if (FOUNDRY_DEBUGGER_SOURCE_GET_CLASS (self)->dup_id)
    return FOUNDRY_DEBUGGER_SOURCE_GET_CLASS (self)->dup_id (self);

  return NULL;
}

char *
foundry_debugger_source_dup_name (FoundryDebuggerSource *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_SOURCE (self), NULL);

  if (FOUNDRY_DEBUGGER_SOURCE_GET_CLASS (self)->dup_name)
    return FOUNDRY_DEBUGGER_SOURCE_GET_CLASS (self)->dup_name (self);

  return NULL;
}

char *
foundry_debugger_source_dup_path (FoundryDebuggerSource *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_SOURCE (self), NULL);

  if (FOUNDRY_DEBUGGER_SOURCE_GET_CLASS (self)->dup_path)
    return FOUNDRY_DEBUGGER_SOURCE_GET_CLASS (self)->dup_path (self);

  return NULL;
}
