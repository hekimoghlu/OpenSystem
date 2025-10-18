/* foundry-file-monitor-event.c
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

#include "foundry-file-monitor-event.h"

struct _FoundryFileMonitorEvent
{
  GObject            parent_instance;
  GFile             *file;
  GFile             *other_file;
  GFileMonitorEvent  event;
};

enum {
  PROP_0,
  PROP_FILE,
  PROP_OTHER_FILE,
  PROP_EVENT,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryFileMonitorEvent, foundry_file_monitor_event, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_file_monitor_event_finalize (GObject *object)
{
  FoundryFileMonitorEvent *self = (FoundryFileMonitorEvent *)object;

  g_clear_object (&self->file);
  g_clear_object (&self->other_file);

  G_OBJECT_CLASS (foundry_file_monitor_event_parent_class)->finalize (object);
}

static void
foundry_file_monitor_event_get_property (GObject    *object,
                                         guint       prop_id,
                                         GValue     *value,
                                         GParamSpec *pspec)
{
  FoundryFileMonitorEvent *self = FOUNDRY_FILE_MONITOR_EVENT (object);

  switch (prop_id)
    {
    case PROP_FILE:
      g_value_take_object (value, foundry_file_monitor_event_dup_file (self));
      break;

    case PROP_OTHER_FILE:
      g_value_take_object (value, foundry_file_monitor_event_dup_other_file (self));
      break;

    case PROP_EVENT:
      g_value_set_enum (value, foundry_file_monitor_event_get_event (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_file_monitor_event_set_property (GObject      *object,
                                         guint         prop_id,
                                         const GValue *value,
                                         GParamSpec   *pspec)
{
  FoundryFileMonitorEvent *self = FOUNDRY_FILE_MONITOR_EVENT (object);

  switch (prop_id)
    {
    case PROP_FILE:
      self->file = g_value_dup_object (value);
      break;

    case PROP_OTHER_FILE:
      self->other_file = g_value_dup_object (value);
      break;

    case PROP_EVENT:
      self->event = g_value_get_enum (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_file_monitor_event_class_init (FoundryFileMonitorEventClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_file_monitor_event_finalize;
  object_class->get_property = foundry_file_monitor_event_get_property;
  object_class->set_property = foundry_file_monitor_event_set_property;

  properties[PROP_FILE] =
    g_param_spec_object ("file", NULL, NULL,
                         G_TYPE_FILE,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_OTHER_FILE] =
    g_param_spec_object ("other-file", NULL, NULL,
                         G_TYPE_FILE,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  properties[PROP_EVENT] =
    g_param_spec_enum ("event", NULL, NULL,
                       G_TYPE_FILE_MONITOR_EVENT,
                       0,
                       (G_PARAM_READWRITE |
                        G_PARAM_CONSTRUCT_ONLY |
                        G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_file_monitor_event_init (FoundryFileMonitorEvent *self)
{
}

FoundryFileMonitorEvent *
foundry_file_monitor_event_new (GFile             *file,
                                GFile             *other_file,
                                GFileMonitorEvent  event)
{
  g_return_val_if_fail (G_IS_FILE (file), NULL);
  g_return_val_if_fail (!other_file || G_IS_FILE (other_file), NULL);

  return g_object_new (FOUNDRY_TYPE_FILE_MONITOR_EVENT,
                       "file", file,
                       "other-file", other_file,
                       "event", event,
                       NULL);
}

/**
 * foundry_file_monitor_event_dup_file:
 * @self: a [class@Foundry.FileMonitorEvent]
 *
 * Returns: (transfer full):
 */
GFile *
foundry_file_monitor_event_dup_file (FoundryFileMonitorEvent *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FILE_MONITOR_EVENT (self), NULL);

  return g_object_ref (self->file);
}

/**
 * foundry_file_monitor_event_dup_other_file:
 * @self: a [class@Foundry.FileMonitorEvent]
 *
 * Returns: (transfer full) (nullable):
 */
GFile *
foundry_file_monitor_event_dup_other_file (FoundryFileMonitorEvent *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FILE_MONITOR_EVENT (self), NULL);

  return self->other_file ? g_object_ref (self->other_file) : NULL;
}

GFileMonitorEvent
foundry_file_monitor_event_get_event (FoundryFileMonitorEvent *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FILE_MONITOR_EVENT (self), 0);

  return self->event;
}
