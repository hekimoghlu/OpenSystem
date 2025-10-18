/* plugin-gdb-mapped-region.c
 *
 * Copyright 2025 Christian Hergert
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include "plugin-gdb-mapped-region.h"

struct _PluginGdbMappedRegion
{
  FoundryDebuggerMappedRegion parent_instance;
  char *path;
  guint64 begin;
  guint64 end;
  guint64 offset;
  guint mode;
};

G_DEFINE_FINAL_TYPE (PluginGdbMappedRegion, plugin_gdb_mapped_region, FOUNDRY_TYPE_DEBUGGER_MAPPED_REGION)

static char *
plugin_gdb_mapped_region_dup_path (FoundryDebuggerMappedRegion *mapped_region)
{
  return g_strdup (PLUGIN_GDB_MAPPED_REGION (mapped_region)->path);
}

static guint
plugin_gdb_mapped_region_get_mode (FoundryDebuggerMappedRegion *mapped_region)
{
  return PLUGIN_GDB_MAPPED_REGION (mapped_region)->mode;
}

static void
plugin_gdb_mapped_region_get_range (FoundryDebuggerMappedRegion *mapped_region,
                                    guint64                     *begin,
                                    guint64                     *end)
{
  if (begin)
    *begin = PLUGIN_GDB_MAPPED_REGION (mapped_region)->begin;

  if (end)
    *end = PLUGIN_GDB_MAPPED_REGION (mapped_region)->end;
}

static guint64
plugin_gdb_mapped_region_get_offset (FoundryDebuggerMappedRegion *mapped_region)
{
  return PLUGIN_GDB_MAPPED_REGION (mapped_region)->offset;
}

static void
plugin_gdb_mapped_region_finalize (GObject *object)
{
  PluginGdbMappedRegion *self = (PluginGdbMappedRegion *)object;

  g_clear_pointer (&self->path, g_free);

  G_OBJECT_CLASS (plugin_gdb_mapped_region_parent_class)->finalize (object);
}

static void
plugin_gdb_mapped_region_class_init (PluginGdbMappedRegionClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);
  FoundryDebuggerMappedRegionClass *mapped_region_class = FOUNDRY_DEBUGGER_MAPPED_REGION_CLASS (klass);

  object_class->finalize = plugin_gdb_mapped_region_finalize;

  mapped_region_class->dup_path = plugin_gdb_mapped_region_dup_path;
  mapped_region_class->get_mode = plugin_gdb_mapped_region_get_mode;
  mapped_region_class->get_range = plugin_gdb_mapped_region_get_range;
  mapped_region_class->get_offset = plugin_gdb_mapped_region_get_offset;
}

static void
plugin_gdb_mapped_region_init (PluginGdbMappedRegion *self)
{
}

FoundryDebuggerMappedRegion *
plugin_gdb_mapped_region_new (guint64     begin,
                              guint64     end,
                              guint64     offset,
                              guint       mode,
                              const char *path)
{
  PluginGdbMappedRegion *self = g_object_new (PLUGIN_TYPE_GDB_MAPPED_REGION, NULL);

  self->begin = begin;
  self->end = end;
  self->offset = offset;
  self->mode = mode;
  self->path = g_strdup (path);

  return FOUNDRY_DEBUGGER_MAPPED_REGION (self);
}
