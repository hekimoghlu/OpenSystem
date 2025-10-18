/* foundry-file-monitor.c
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

#include "foundry-file-monitor.h"
#include "foundry-file-monitor-event.h"

struct _FoundryFileMonitor
{
  GObject       parent_instance;
  GFile        *file;
  GFileMonitor *monitor;
  DexChannel   *channel;
};

enum {
  PROP_0,
  PROP_FILE,
  N_PROPS
};

G_DEFINE_FINAL_TYPE (FoundryFileMonitor, foundry_file_monitor, G_TYPE_OBJECT)

static GParamSpec *properties[N_PROPS];

static void
foundry_file_monitor_finalize (GObject *object)
{
  FoundryFileMonitor *self = (FoundryFileMonitor *)object;

  dex_channel_close_send (self->channel);

  if (self->monitor)
    g_file_monitor_cancel (self->monitor);

  g_clear_object (&self->file);
  g_clear_object (&self->monitor);

  dex_clear (&self->channel);

  G_OBJECT_CLASS (foundry_file_monitor_parent_class)->finalize (object);
}

static void
foundry_file_monitor_get_property (GObject    *object,
                                   guint       prop_id,
                                   GValue     *value,
                                   GParamSpec *pspec)
{
  FoundryFileMonitor *self = FOUNDRY_FILE_MONITOR (object);

  switch (prop_id)
    {
    case PROP_FILE:
      g_value_take_object (value, foundry_file_monitor_dup_file (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_file_monitor_set_property (GObject      *object,
                                   guint         prop_id,
                                   const GValue *value,
                                   GParamSpec   *pspec)
{
  FoundryFileMonitor *self = FOUNDRY_FILE_MONITOR (object);

  switch (prop_id)
    {
    case PROP_FILE:
      self->file = g_value_dup_object (value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    }
}

static void
foundry_file_monitor_class_init (FoundryFileMonitorClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = foundry_file_monitor_finalize;
  object_class->get_property = foundry_file_monitor_get_property;
  object_class->set_property = foundry_file_monitor_set_property;

  properties[PROP_FILE] =
    g_param_spec_object ("file", NULL, NULL,
                         G_TYPE_FILE,
                         (G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS));

  g_object_class_install_properties (object_class, N_PROPS, properties);
}

static void
foundry_file_monitor_init (FoundryFileMonitor *self)
{
  self->channel = dex_channel_new (0);
}

static void
foundry_file_monitor_changed_cb (FoundryFileMonitor *self,
                                 GFile              *file,
                                 GFile              *other_file,
                                 GFileMonitorEvent   event,
                                 GFileMonitor       *monitor)
{
  g_autoptr(FoundryFileMonitorEvent) object = NULL;
  g_autoptr(DexFuture) future = NULL;

  g_assert (FOUNDRY_IS_FILE_MONITOR (self));
  g_assert (G_IS_FILE (file));
  g_assert (!other_file || G_IS_FILE (other_file));
  g_assert (G_IS_FILE_MONITOR (monitor));

  if (self->channel == NULL || !dex_channel_can_send (self->channel))
    return;

  object = foundry_file_monitor_event_new (file, other_file, event);
  future = dex_future_new_take_object (g_steal_pointer (&object));

  dex_future_disown (dex_channel_send (self->channel, g_steal_pointer (&future)));
}

FoundryFileMonitor *
foundry_file_monitor_new (GFile   *file,
                          GError **error)
{
  g_autoptr(FoundryFileMonitor) self = NULL;

  g_return_val_if_fail (G_IS_FILE (file), NULL);

  self = g_object_new (FOUNDRY_TYPE_FILE_MONITOR,
                       "file", file,
                       NULL);

  if (!(self->monitor = g_file_monitor_directory (file, 0, NULL, error)))
    return NULL;

  g_signal_connect_object (self->monitor,
                           "changed",
                           G_CALLBACK (foundry_file_monitor_changed_cb),
                           self,
                           G_CONNECT_SWAPPED);

  return g_steal_pointer (&self);
}

void
foundry_file_monitor_cancel (FoundryFileMonitor *self)
{
  g_return_if_fail (FOUNDRY_IS_FILE_MONITOR (self));

  if (self->monitor != NULL)
    g_file_monitor_cancel (self->monitor);

  if (self->channel != NULL)
    dex_channel_close_send (self->channel);
}

/**
 * foundry_file_monitor_dup_file:
 * @self: a [class@Foundry.FileMonitor]
 *
 * Returns: (transfer full):
 */
GFile *
foundry_file_monitor_dup_file (FoundryFileMonitor *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FILE_MONITOR (self), NULL);

  return g_object_ref (self->file);
}

/**
 * foundry_file_monitor_next:
 * @self: a [class@Foundry.FileMonitor]
 *
 * Returns: (transfer full):
 */
DexFuture *
foundry_file_monitor_next (FoundryFileMonitor *self)
{
  g_return_val_if_fail (FOUNDRY_IS_FILE_MONITOR (self), NULL);

  if (self->channel != NULL)
    return dex_channel_receive (self->channel);

  return dex_future_new_reject (G_IO_ERROR,
                                G_IO_ERROR_CANCELLED,
                                "Monitoring cancelled");
}
