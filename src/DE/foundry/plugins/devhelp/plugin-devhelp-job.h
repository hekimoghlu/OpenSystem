/* plugin-devhelp-job.h
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

#pragma once

#include <gio/gio.h>

G_BEGIN_DECLS

#define PLUGIN_TYPE_DEVHELP_JOB (plugin_devhelp_job_get_type())

G_DECLARE_FINAL_TYPE (PluginDevhelpJob, plugin_devhelp_job, PLUGIN, DEVHELP_JOB, GObject)

PluginDevhelpJob *plugin_devhelp_job_new          (void);
char             *plugin_devhelp_job_dup_title    (PluginDevhelpJob *self);
void              plugin_devhelp_job_set_title    (PluginDevhelpJob *self,
                                                   const char       *title);
char             *plugin_devhelp_job_dup_subtitle (PluginDevhelpJob *self);
void              plugin_devhelp_job_set_subtitle (PluginDevhelpJob *self,
                                                   const char       *subtitle);
double            plugin_devhelp_job_get_fraction (PluginDevhelpJob *self);
void              plugin_devhelp_job_set_fraction (PluginDevhelpJob *self,
                                                   double            fraction);
void              plugin_devhelp_job_complete     (PluginDevhelpJob *self);

typedef struct _PluginDevhelpJob PluginDevhelpJobMonitor;

static inline void
_plugin_devhelp_job_monitor_cleanup_func (PluginDevhelpJobMonitor *monitor)
{
  plugin_devhelp_job_complete (monitor);
  g_object_unref (monitor);
}

G_DEFINE_AUTOPTR_CLEANUP_FUNC (PluginDevhelpJobMonitor, _plugin_devhelp_job_monitor_cleanup_func)

G_END_DECLS
