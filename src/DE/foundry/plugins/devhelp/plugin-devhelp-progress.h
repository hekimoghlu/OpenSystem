/* plugin-devhelp-progress.h
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

#include "plugin-devhelp-job.h"

G_BEGIN_DECLS

#define PLUGIN_TYPE_DEVHELP_PROGRESS (plugin_devhelp_progress_get_type())

G_DECLARE_FINAL_TYPE (PluginDevhelpProgress, plugin_devhelp_progress, PLUGIN, DEVHELP_PROGRESS, GObject)

PluginDevhelpProgress *plugin_devhelp_progress_new          (void);
PluginDevhelpJob      *plugin_devhelp_progress_begin_job    (PluginDevhelpProgress *self);
double                 plugin_devhelp_progress_get_fraction (PluginDevhelpProgress *self);
void                   plugin_devhelp_progress_done         (PluginDevhelpProgress *self);

G_END_DECLS
