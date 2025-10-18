/* foundry-debugger-mapped-region.c
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

enum {
  PROP_0,
  PROP_BEGIN_ADDRESS,
  PROP_END_ADDRESS,
  PROP_MODE,
  PROP_OFFSET,
  PROP_PATH,
  N_PROPS
};

G_DEFINE_ABSTRACT_TYPE (FoundryDebuggerMappedRegion, foundry_debugger_mapped_region, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_debugger_mapped_region_get_property (GObject    *object,
                                             guint       prop_id,
                                             GValue     *value,
                                             GParamSpec *pspec)
{
  FoundryDebuggerMappedRegion *self = FOUNDRY_DEBUGGER_MAPPED_REGION (object);

  switch (prop_id)
    {
    case PROP_MODE:
      g_value_set_uint (value, foundry_debugger_mapped_region_get_mode (self));
      break;

    case PROP_BEGIN_ADDRESS:
    case PROP_END_ADDRESS:
      {
        guint64 begin, end;
        foundry_debugger_mapped_region_get_range (self, &begin, &end);
        g_value_set_uint64 (value, prop_id == PROP_BEGIN_ADDRESS ? begin : end);
        break;
      }

    case PROP_OFFSET:
      g_value_set_uint64 (value, foundry_debugger_mapped_region_get_offset (self));
      break;

    case PROP_PATH:
      g_value_take_string (value, foundry_debugger_mapped_region_dup_path (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_debugger_mapped_region_class_init (FoundryDebuggerMappedRegionClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = foundry_debugger_mapped_region_get_property;

  properties[PROP_MODE] =
    g_param_spec_uint ("mode", NULL, NULL,
                       0, G_MAXUINT, 0,
                       (G_PARAM_READABLE |
                        G_PARAM_STATIC_STRINGS));

  properties[PROP_BEGIN_ADDRESS] =
    g_param_spec_uint64 ("begin-address", NULL, NULL,
                         0, G_MAXUINT64, 0,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_END_ADDRESS] =
    g_param_spec_uint64 ("end-address", NULL, NULL,
                         0, G_MAXUINT64, 0,
                         (G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS));

  /**
   * FoundryDebuggerMappedRegion:offset:
   *
   * The offset within `path` where the mapping originates.
   *
   * Since: 1.1
   */
  properties[PROP_OFFSET] =
    g_param_spec_uint64 ("offset", NULL, NULL,
                         0, G_MAXUINT64, 0,
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
foundry_debugger_mapped_region_init (FoundryDebuggerMappedRegion *self)
{
}

char *
foundry_debugger_mapped_region_dup_path (FoundryDebuggerMappedRegion *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_MAPPED_REGION (self), NULL);

  if (FOUNDRY_DEBUGGER_MAPPED_REGION_GET_CLASS (self)->dup_path)
    return FOUNDRY_DEBUGGER_MAPPED_REGION_GET_CLASS (self)->dup_path (self);

  return NULL;
}

guint
foundry_debugger_mapped_region_get_mode (FoundryDebuggerMappedRegion *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_MAPPED_REGION (self), 0);

  if (FOUNDRY_DEBUGGER_MAPPED_REGION_GET_CLASS (self)->get_mode)
    return FOUNDRY_DEBUGGER_MAPPED_REGION_GET_CLASS (self)->get_mode (self);

  return 0;
}

void
foundry_debugger_mapped_region_get_range (FoundryDebuggerMappedRegion *self,
                                          guint64                     *begin_address,
                                          guint64                     *end_address)
{
  guint64 dummy1 = 0;
  guint64 dummy2 = 0;

  if (begin_address == NULL)
    begin_address = &dummy1;

  if (end_address == NULL)
    end_address = &dummy2;

  if (FOUNDRY_DEBUGGER_MAPPED_REGION_GET_CLASS (self)->get_range)
    return FOUNDRY_DEBUGGER_MAPPED_REGION_GET_CLASS (self)->get_range (self, begin_address, end_address);

  *begin_address = 0;
  *end_address = 0;
}

guint64
foundry_debugger_mapped_region_get_offset (FoundryDebuggerMappedRegion *self)
{
  g_return_val_if_fail (FOUNDRY_IS_DEBUGGER_MAPPED_REGION (self), 0);

  if (FOUNDRY_DEBUGGER_MAPPED_REGION_GET_CLASS (self)->get_offset)
    return FOUNDRY_DEBUGGER_MAPPED_REGION_GET_CLASS (self)->get_offset (self);

  return 0;
}
