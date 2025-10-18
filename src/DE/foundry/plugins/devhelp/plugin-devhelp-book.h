/* plugin-devhelp-book.h
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

#include <gom/gom.h>
#include <libdex.h>

G_BEGIN_DECLS

#define PLUGIN_TYPE_DEVHELP_BOOK (plugin_devhelp_book_get_type())

G_DECLARE_FINAL_TYPE (PluginDevhelpBook, plugin_devhelp_book, PLUGIN, DEVHELP_BOOK, GomResource)

gint64      plugin_devhelp_book_get_id          (PluginDevhelpBook *self);
void        plugin_devhelp_book_set_id          (PluginDevhelpBook *self,
                                                 gint64       id);
gint64      plugin_devhelp_book_get_sdk_id      (PluginDevhelpBook *self);
void        plugin_devhelp_book_set_sdk_id      (PluginDevhelpBook *self,
                                                 gint64       sdk_id);
const char *plugin_devhelp_book_get_etag        (PluginDevhelpBook *self);
void        plugin_devhelp_book_set_etag        (PluginDevhelpBook *self,
                                                 const char  *etag);
const char *plugin_devhelp_book_get_title       (PluginDevhelpBook *self);
void        plugin_devhelp_book_set_title       (PluginDevhelpBook *self,
                                                 const char  *title);
const char *plugin_devhelp_book_get_uri         (PluginDevhelpBook *self);
void        plugin_devhelp_book_set_uri         (PluginDevhelpBook *self,
                                                 const char  *uri);
const char *plugin_devhelp_book_get_default_uri (PluginDevhelpBook *self);
void        plugin_devhelp_book_set_default_uri (PluginDevhelpBook *self,
                                                 const char  *default_uri);
const char *plugin_devhelp_book_get_online_uri  (PluginDevhelpBook *self);
void        plugin_devhelp_book_set_online_uri  (PluginDevhelpBook *self,
                                                 const char  *online_uri);
const char *plugin_devhelp_book_get_language    (PluginDevhelpBook *self);
void        plugin_devhelp_book_set_language    (PluginDevhelpBook *self,
                                                 const char  *language);
DexFuture  *plugin_devhelp_book_list_headings   (PluginDevhelpBook *self);
DexFuture  *plugin_devhelp_book_list_alternates (PluginDevhelpBook *self);
DexFuture  *plugin_devhelp_book_find_sdk        (PluginDevhelpBook *self);

G_END_DECLS

