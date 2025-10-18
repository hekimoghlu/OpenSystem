/* foundry-command.h
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

#include <glib-object.h>

#include "foundry-contextual.h"
#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_COMMAND          (foundry_command_get_type())
#define FOUNDRY_TYPE_COMMAND_LOCALITY (foundry_command_locality_get_type())

/**
 * FoundryCommandLocality:
 * %FOUNDRY_COMMAND_LOCALITY_SUBPROCESS: run as a subprocess of builder
 * %FOUNDRY_COMMAND_LOCALITY_HOST: run on the host system, possibly bypassing container
 * %FOUNDRY_COMMAND_LOCALITY_BUILD_PIPELINE: run from build pipeline
 * %FOUNDRY_COMMAND_LOCALITY_APPLICATION: run like a target application
 */
typedef enum _FoundryCommandLocality
{
  FOUNDRY_COMMAND_LOCALITY_SUBPROCESS = 0,
  FOUNDRY_COMMAND_LOCALITY_HOST,
  FOUNDRY_COMMAND_LOCALITY_BUILD_PIPELINE,
  FOUNDRY_COMMAND_LOCALITY_APPLICATION,

  /* Not part of ABI */
  FOUNDRY_COMMAND_LOCALITY_LAST,
} FoundryCommandLocality;

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_DERIVABLE_TYPE (FoundryCommand, foundry_command, FOUNDRY, COMMAND, FoundryContextual)

struct _FoundryCommandClass
{
  FoundryContextualClass parent_class;

  gboolean   (*can_default) (FoundryCommand            *self,
                             guint                     *priority);
  DexFuture *(*prepare)     (FoundryCommand            *self,
                             FoundryBuildPipeline      *pipeline,
                             FoundryProcessLauncher    *launcher,
                             FoundryBuildPipelinePhase  phase);

  /*< private >*/
  gpointer _reserved[6];
};

FOUNDRY_AVAILABLE_IN_ALL
GType                    foundry_command_locality_get_type (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
FoundryCommand          *foundry_command_new               (FoundryContext            *context);
FOUNDRY_AVAILABLE_IN_ALL
char                    *foundry_command_dup_id            (FoundryCommand            *self);
FOUNDRY_AVAILABLE_IN_ALL
void                     foundry_command_set_id            (FoundryCommand            *self,
                                                            const char                *id);
FOUNDRY_AVAILABLE_IN_ALL
char                   **foundry_command_dup_argv          (FoundryCommand            *self);
FOUNDRY_AVAILABLE_IN_ALL
void                     foundry_command_set_argv          (FoundryCommand            *self,
                                                            const char * const        *argv);
FOUNDRY_AVAILABLE_IN_ALL
char                    *foundry_command_dup_cwd           (FoundryCommand            *self);
FOUNDRY_AVAILABLE_IN_ALL
void                     foundry_command_set_cwd           (FoundryCommand            *self,
                                                            const char                *cwd);
FOUNDRY_AVAILABLE_IN_ALL
char                   **foundry_command_dup_environ       (FoundryCommand            *self);
FOUNDRY_AVAILABLE_IN_ALL
void                     foundry_command_set_environ       (FoundryCommand            *self,
                                                            const char * const        *environ);
FOUNDRY_AVAILABLE_IN_ALL
FoundryCommandLocality   foundry_command_get_locality      (FoundryCommand            *self);
FOUNDRY_AVAILABLE_IN_ALL
void                     foundry_command_set_locality      (FoundryCommand            *self,
                                                            FoundryCommandLocality     locality);
FOUNDRY_AVAILABLE_IN_ALL
char                    *foundry_command_dup_name          (FoundryCommand            *self);
FOUNDRY_AVAILABLE_IN_ALL
void                     foundry_command_set_name          (FoundryCommand            *self,
                                                            const char                *id);
FOUNDRY_AVAILABLE_IN_ALL
gboolean                 foundry_command_can_default       (FoundryCommand            *self,
                                                            guint                     *priority);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture               *foundry_command_prepare           (FoundryCommand            *self,
                                                            FoundryBuildPipeline      *pipeline,
                                                            FoundryProcessLauncher    *launcher,
                                                            FoundryBuildPipelinePhase  phase) G_GNUC_WARN_UNUSED_RESULT;

G_END_DECLS
