/* foundry-git-uri.h
 *
 * Copyright 2015-2025 Christian Hergert <chergert@redhat.com>
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

#include <glib-object.h>

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_GIT_URI (foundry_git_uri_get_type())

typedef struct _FoundryGitUri FoundryGitUri;

FOUNDRY_AVAILABLE_IN_ALL
GType          foundry_git_uri_get_type       (void);
FOUNDRY_AVAILABLE_IN_ALL
FoundryGitUri *foundry_git_uri_new            (const char          *uri);
FOUNDRY_AVAILABLE_IN_ALL
FoundryGitUri *foundry_git_uri_ref            (FoundryGitUri       *self);
FOUNDRY_AVAILABLE_IN_ALL
void           foundry_git_uri_unref          (FoundryGitUri       *self);
FOUNDRY_AVAILABLE_IN_ALL
const char    *foundry_git_uri_get_scheme     (const FoundryGitUri *self);
FOUNDRY_AVAILABLE_IN_ALL
const char    *foundry_git_uri_get_user       (const FoundryGitUri *self);
FOUNDRY_AVAILABLE_IN_ALL
const char    *foundry_git_uri_get_host       (const FoundryGitUri *self);
FOUNDRY_AVAILABLE_IN_ALL
guint          foundry_git_uri_get_port       (const FoundryGitUri *self);
FOUNDRY_AVAILABLE_IN_ALL
const char    *foundry_git_uri_get_path       (const FoundryGitUri *self);
FOUNDRY_AVAILABLE_IN_ALL
void           foundry_git_uri_set_scheme     (FoundryGitUri       *self,
                                               const char          *scheme);
FOUNDRY_AVAILABLE_IN_ALL
void           foundry_git_uri_set_user       (FoundryGitUri       *self,
                                               const char          *user);
FOUNDRY_AVAILABLE_IN_ALL
void           foundry_git_uri_set_host       (FoundryGitUri       *self,
                                               const char          *host);
FOUNDRY_AVAILABLE_IN_ALL
void           foundry_git_uri_set_port       (FoundryGitUri       *self,
                                               guint                port);
FOUNDRY_AVAILABLE_IN_ALL
void           foundry_git_uri_set_path       (FoundryGitUri       *self,
                                               const char          *path);
FOUNDRY_AVAILABLE_IN_ALL
char          *foundry_git_uri_to_string      (const FoundryGitUri *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean       foundry_git_uri_is_valid       (const char          *uri_string);
FOUNDRY_AVAILABLE_IN_ALL
char          *foundry_git_uri_get_clone_name (const FoundryGitUri *self);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (FoundryGitUri, foundry_git_uri_unref)

G_END_DECLS
