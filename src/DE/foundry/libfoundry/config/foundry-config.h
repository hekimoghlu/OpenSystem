/* foundry-config.h
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
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

#include "foundry-contextual.h"
#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_CONFIG   (foundry_config_get_type())
#define FOUNDRY_TYPE_LOCALITY (foundry_locality_get_type())

typedef enum _FoundryLocality
{
  FOUNDRY_LOCALITY_BUILD,
  FOUNDRY_LOCALITY_RUN,
  FOUNDRY_LOCALITY_TOOL,

  FOUNDRY_LOCALITY_LAST
} FoundryLocality;

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryConfig, foundry_config, FOUNDRY, CONFIG, FoundryContextual)

struct _FoundryConfigClass
{
  FoundryContextualClass parent_class;

  DexFuture       *(*resolve_sdk)         (FoundryConfig        *self,
                                           FoundryDevice        *device);
  char           **(*dup_environ)         (FoundryConfig        *self,
                                           FoundryLocality       locality);
  char            *(*dup_build_system)    (FoundryConfig        *self);
  gboolean         (*can_default)         (FoundryConfig        *self,
                                           guint                *priority);
  char            *(*dup_builddir)        (FoundryConfig        *self,
                                           FoundryBuildPipeline *pipeline);
  char           **(*dup_config_opts)     (FoundryConfig        *self);
  FoundryCommand  *(*dup_default_command) (FoundryConfig        *self);

  /*< private >*/
  gpointer _reserved[9];
};

FOUNDRY_AVAILABLE_IN_ALL
GType            foundry_locality_get_type          (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_config_get_active          (FoundryConfig        *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean         foundry_config_can_default         (FoundryConfig        *self,
                                                     guint                *priority);
FOUNDRY_AVAILABLE_IN_ALL
char            *foundry_config_dup_id              (FoundryConfig        *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_config_set_id              (FoundryConfig        *self,
                                                     const char           *id);
FOUNDRY_AVAILABLE_IN_ALL
char            *foundry_config_dup_build_system    (FoundryConfig        *self);
FOUNDRY_AVAILABLE_IN_ALL
char            *foundry_config_dup_name            (FoundryConfig        *self);
FOUNDRY_AVAILABLE_IN_ALL
void             foundry_config_set_name            (FoundryConfig        *self,
                                                     const char           *name);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture       *foundry_config_resolve_sdk         (FoundryConfig        *self,
                                                     FoundryDevice        *device);
FOUNDRY_AVAILABLE_IN_ALL
char           **foundry_config_dup_environ         (FoundryConfig        *self,
                                                     FoundryLocality       locality);
FOUNDRY_AVAILABLE_IN_ALL
char            *foundry_config_dup_builddir        (FoundryConfig        *self,
                                                     FoundryBuildPipeline *pipeline);
FOUNDRY_AVAILABLE_IN_ALL
char           **foundry_config_dup_config_opts     (FoundryConfig        *self);
FOUNDRY_AVAILABLE_IN_ALL
FoundryCommand  *foundry_config_dup_default_command (FoundryConfig        *self);

G_END_DECLS
