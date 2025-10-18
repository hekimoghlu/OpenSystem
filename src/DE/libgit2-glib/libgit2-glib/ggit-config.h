/*
 * ggit-config.h
 * This file is part of libgit2-glib
 *
 * Copyright (C) 2012 - Jesse van den Kieboom
 *
 * libgit2-glib is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libgit2-glib is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libgit2-glib. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __GGIT_CONFIG_H__
#define __GGIT_CONFIG_H__

#include <glib-object.h>
#include <gio/gio.h>
#include <git2.h>

#include <libgit2-glib/ggit-types.h>
#include "ggit-native.h"

G_BEGIN_DECLS

#define GGIT_TYPE_CONFIG (ggit_config_get_type ())
G_DECLARE_FINAL_TYPE (GgitConfig, ggit_config, GGIT, CONFIG, GgitNative)

GgitConfig  *ggit_config_new           (void);

GgitConfig  *ggit_config_new_default   (GError                 **error);

GgitConfig  *ggit_config_new_from_file (GFile                   *file,
                                        GError                 **error);

GFile       *ggit_config_find_global   (void);
GFile       *ggit_config_find_system   (void);

GgitConfig  *ggit_config_open_level    (GgitConfig               *config,
                                        GgitConfigLevel           level,
                                        GError                  **error);

void         ggit_config_add_file      (GgitConfig               *config,
                                        GFile                    *file,
                                        GgitConfigLevel           level,
                                        gboolean                  force,
                                        GError                  **error);

gint32       ggit_config_get_int32     (GgitConfig               *config,
                                        const gchar              *name,
                                        GError                  **error);

gboolean     ggit_config_set_int32     (GgitConfig               *config,
                                        const gchar              *name,
                                        gint32                    value,
                                        GError                  **error);

gint64       ggit_config_get_int64     (GgitConfig               *config,
                                        const gchar              *name,
                                        GError                  **error);

gboolean     ggit_config_set_int64     (GgitConfig               *config,
                                        const gchar              *name,
                                        gint64                    value,
                                        GError                  **error);

gboolean     ggit_config_get_bool      (GgitConfig               *config,
                                        const gchar              *name,
                                        GError                  **error);

gboolean     ggit_config_set_bool      (GgitConfig               *config,
                                        const gchar              *name,
                                        gboolean                  value,
                                        GError                  **error);

const gchar *ggit_config_get_string    (GgitConfig               *config,
                                        const gchar              *name,
                                        GError                  **error);

gboolean     ggit_config_set_string    (GgitConfig               *config,
                                        const gchar              *name,
                                        const gchar              *value,
                                        GError                  **error);

GgitConfigEntry *
             ggit_config_get_entry     (GgitConfig               *config,
                                        const gchar              *name,
                                        GError                  **error);

gboolean     ggit_config_delete_entry  (GgitConfig               *config,
                                        const gchar              *name,
                                        GError                  **error);

gboolean     ggit_config_foreach       (GgitConfig               *config,
                                        GgitConfigCallback        callback,
                                        gpointer                  user_data,
                                        GError                  **error);

gchar       *ggit_config_match         (GgitConfig               *config,
                                        GRegex                   *regex,
                                        GMatchInfo              **match_info,
                                        GError                  **error);

gboolean     ggit_config_match_foreach (GgitConfig               *config,
                                        GRegex                   *regex,
                                        GgitConfigMatchCallback   callback,
                                        gpointer                  user_data,
                                        GError                  **error);

GgitConfig  *ggit_config_snapshot      (GgitConfig               *config,
                                        GError                  **error);

GgitConfig  *_ggit_config_wrap         (git_config               *config);

G_END_DECLS

#endif /* __GGIT_CONFIG_H__ */

/* ex:set ts=8 noet: */
