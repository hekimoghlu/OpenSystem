/* plugin-doap-file.h
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

#include "plugin-doap-person.h"

G_BEGIN_DECLS

#define PLUGIN_DOAP_FILE_ERROR (plugin_doap_file_error_quark())
#define PLUGIN_TYPE_DOAP_FILE  (plugin_doap_file_get_type())

G_DECLARE_FINAL_TYPE (PluginDoapFile, plugin_doap_file, PLUGIN, DOAP_FILE, GObject)

typedef enum
{
  PLUGIN_DOAP_FILE_ERROR_INVALID_FORMAT = 1,
} PluginDoapFileError;

PluginDoapFile  *plugin_doap_file_new               (void);
GQuark           plugin_doap_file_error_quark       (void);
gboolean         plugin_doap_file_load_from_file    (PluginDoapFile  *self,
                                                     GFile           *file,
                                                     GCancellable    *cancellable,
                                                     GError         **error);
gboolean         plugin_doap_file_load_from_data    (PluginDoapFile  *self,
                                                     const char      *data,
                                                     gsize            length,
                                                     GError         **error);
gboolean         plugin_doap_file_load_from_bytes   (PluginDoapFile  *self,
                                                     GBytes          *bytes,
                                                     GError         **error);
const char      *plugin_doap_file_get_name          (PluginDoapFile  *self);
const char      *plugin_doap_file_get_shortdesc     (PluginDoapFile  *self);
const char      *plugin_doap_file_get_description   (PluginDoapFile  *self);
const char      *plugin_doap_file_get_bug_database  (PluginDoapFile  *self);
const char      *plugin_doap_file_get_download_page (PluginDoapFile  *self);
const char      *plugin_doap_file_get_homepage      (PluginDoapFile  *self);
const char      *plugin_doap_file_get_category      (PluginDoapFile  *self);
char           **plugin_doap_file_get_languages     (PluginDoapFile  *self);
GList           *plugin_doap_file_get_maintainers   (PluginDoapFile  *self);

G_END_DECLS

