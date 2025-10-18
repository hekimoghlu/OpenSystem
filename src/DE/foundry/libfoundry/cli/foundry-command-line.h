/* foundry-command-line.h
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

#include <gio/gio.h>

#include "foundry-context.h"
#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_COMMAND_LINE             (foundry_command_line_get_type())
#define FOUNDRY_TYPE_OBJECT_SERIALIZER_FORMAT (foundry_object_serializer_format_get_type())
#define FOUNDRY_COMMAND_LINE_ERROR            (foundry_command_line_error_quark())

/**
 * FoundryCommandLineError:
 * %FOUNDRY_COMMAND_LINE_ERROR_RUN_LOCAL: indicate that the command should be run
 *   on the client side rather than in the parent process.
 */
typedef enum _FoundryCommandLineError
{
  FOUNDRY_COMMAND_LINE_ERROR_RUN_LOCAL = 1,
} FoundryCommandLineError;

/**
 * FoundryObjectSerializerEntry:
 * @property: the property name
 * @heading: the column title
 *
 * Used to determine what properties to serialize in command line data.
 */
typedef struct _FoundryObjectSerializerEntry
{
  const char *property;
  const char *heading;
} FoundryObjectSerializerEntry;

typedef enum _FoundryObjectSerializerFormat
{
  FOUNDRY_OBJECT_SERIALIZER_FORMAT_TEXT,
  FOUNDRY_OBJECT_SERIALIZER_FORMAT_JSON,
} FoundryObjectSerializerFormat;

FOUNDRY_AVAILABLE_IN_ALL
GType                         foundry_object_serializer_format_get_type (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
FoundryObjectSerializerFormat foundry_object_serializer_format_parse    (const char *string);

FOUNDRY_AVAILABLE_IN_ALL
FOUNDRY_DECLARE_INTERNAL_TYPE (FoundryCommandLine, foundry_command_line, FOUNDRY, COMMAND_LINE, GObject)

FOUNDRY_AVAILABLE_IN_ALL
GQuark                foundry_command_line_error_quark        (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
FoundryCommandLine   *foundry_command_line_new                (void);
FOUNDRY_AVAILABLE_IN_ALL
const char           *foundry_command_line_getenv             (FoundryCommandLine                 *self,
                                                               const char                         *name);
FOUNDRY_AVAILABLE_IN_ALL
char                 *foundry_command_line_get_directory      (FoundryCommandLine                 *self);
FOUNDRY_AVAILABLE_IN_ALL
char                **foundry_command_line_get_environ        (FoundryCommandLine                 *self);
FOUNDRY_AVAILABLE_IN_ALL
void                  foundry_command_line_print              (FoundryCommandLine                 *self,
                                                               const char                         *format,
                                                               ...) G_GNUC_PRINTF (2, 3);
FOUNDRY_AVAILABLE_IN_ALL
void                  foundry_command_line_printerr           (FoundryCommandLine                 *self,
                                                               const char                         *format,
                                                               ...) G_GNUC_PRINTF (2, 3);
FOUNDRY_AVAILABLE_IN_ALL
void                  foundry_command_line_print_list         (FoundryCommandLine                 *self,
                                                               GListModel                         *model,
                                                               const FoundryObjectSerializerEntry *entries,
                                                               FoundryObjectSerializerFormat       format,
                                                               GType                               expected_type);
FOUNDRY_AVAILABLE_IN_ALL
void                  foundry_command_line_print_object       (FoundryCommandLine                 *self,
                                                               GObject                            *object,
                                                               const FoundryObjectSerializerEntry *entries,
                                                               FoundryObjectSerializerFormat       format);
FOUNDRY_AVAILABLE_IN_ALL
gboolean              foundry_command_line_isatty             (FoundryCommandLine                 *self);
FOUNDRY_AVAILABLE_IN_ALL
gboolean              foundry_command_line_is_remote          (FoundryCommandLine                 *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture            *foundry_command_line_run                (FoundryCommandLine                 *self,
                                                               const char * const                 *argv) G_GNUC_WARN_UNUSED_RESULT;
FOUNDRY_AVAILABLE_IN_ALL
int                   foundry_command_line_get_stdin          (FoundryCommandLine                 *self);
FOUNDRY_AVAILABLE_IN_ALL
int                   foundry_command_line_get_stdout         (FoundryCommandLine                 *self);
FOUNDRY_AVAILABLE_IN_ALL
int                   foundry_command_line_get_stderr         (FoundryCommandLine                 *self);
FOUNDRY_AVAILABLE_IN_ALL
void                  foundry_command_line_set_progress       (FoundryCommandLine                 *self,
                                                               guint                               progress);
FOUNDRY_AVAILABLE_IN_ALL
void                  foundry_command_line_clear_progress     (FoundryCommandLine                 *self);
FOUNDRY_AVAILABLE_IN_ALL
void                  foundry_command_line_set_title          (FoundryCommandLine                 *self,
                                                               const char                         *title);
FOUNDRY_AVAILABLE_IN_ALL
FoundryAuthProvider  *foundry_command_line_dup_auth_provider  (FoundryCommandLine                 *self);
FOUNDRY_AVAILABLE_IN_ALL
GFile                *foundry_command_line_build_file_for_arg (FoundryCommandLine                 *self,
                                                               const char                         *arg);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture            *foundry_command_line_request_input      (FoundryCommandLine                 *self,
                                                               FoundryInput                       *input);

G_END_DECLS
