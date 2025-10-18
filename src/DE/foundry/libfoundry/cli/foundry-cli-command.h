/* foundry-cli-command.h
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

#include <libdex.h>

#include "foundry-types.h"
#include "foundry-version-macros.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_CLI_OPTIONS (foundry_cli_options_get_type())

typedef struct _FoundryCliOptions FoundryCliOptions;
typedef struct _FoundryCliCommand FoundryCliCommand;

struct _FoundryCliCommand
{
  /* `{0}` terminated array of GOptionEntry used to both parse options
   * into a FoundryCliOptions as well as auto-complete from shell.
   */
  GOptionEntry *options;

  /* Called to run the command. All parent commands will have had their
   * options parsed into @options and are available for use. The command
   * name will be compressed into a "foundry-parent-child" as argv[0] and
   * all options will be preparsed into @options. Any remaining options
   * are placed starting from argv[1].
   *
   * This function is run from a fiber so you may await completion of
   * futures natively from the run callback.
   */
  int (*run) (FoundryCommandLine *command_line,
              const char * const *argv,
              FoundryCliOptions  *options,
              DexCancellable     *cancellable);

  /* Optional callback to setup the GOptionContext when incrementally
   * parsing the callback tree. This allows FoundryCliCommand to tweak
   * various aspects of parsing.
   */
  void (*prepare) (GOptionContext *context);

  /* Complete an option arg if otherwise unhandled. These should not do
   * IO which is why there is no current-directory provided. Files should
   * be completed as __FOUNDRY_FILE.
   */
  char **(*complete) (FoundryCommandLine *command_line,
                      const char         *command,
                      const GOptionEntry *entry,
                      FoundryCliOptions  *options,
                      const char * const *argv,
                      const char         *current);

  /* If specified, will be used for translation of entries */
  const char *gettext_package;

  /* If specified, the description for command */
  const char *description;
};

FOUNDRY_AVAILABLE_IN_ALL
GType               foundry_cli_options_get_type           (void) G_GNUC_CONST;
FOUNDRY_AVAILABLE_IN_ALL
FoundryCliOptions  *foundry_cli_options_new                (void);
FOUNDRY_AVAILABLE_IN_ALL
FoundryCliOptions  *foundry_cli_options_ref                (FoundryCliOptions   *self);
FOUNDRY_AVAILABLE_IN_ALL
void                foundry_cli_options_unref              (FoundryCliOptions   *self);
FOUNDRY_AVAILABLE_IN_ALL
DexFuture          *foundry_cli_options_load_context       (FoundryCliOptions   *self,
                                                            FoundryCommandLine  *command_line);
FOUNDRY_AVAILABLE_IN_ALL
gboolean            foundry_cli_options_help               (FoundryCliOptions   *self);
FOUNDRY_AVAILABLE_IN_ALL
void                foundry_cli_options_set_string         (FoundryCliOptions   *self,
                                                            const char          *key,
                                                            const char          *value);
FOUNDRY_AVAILABLE_IN_ALL
const char         *foundry_cli_options_get_string         (FoundryCliOptions   *self,
                                                            const char          *key);
FOUNDRY_AVAILABLE_IN_ALL
void                foundry_cli_options_set_string_array   (FoundryCliOptions   *self,
                                                            const char          *key,
                                                            const char * const  *value);
FOUNDRY_AVAILABLE_IN_ALL
const char * const *foundry_cli_options_get_string_array   (FoundryCliOptions   *self,
                                                            const char          *key);
FOUNDRY_AVAILABLE_IN_ALL
void                foundry_cli_options_set_filename       (FoundryCliOptions   *self,
                                                            const char          *key,
                                                            const char          *value);
FOUNDRY_AVAILABLE_IN_ALL
const char         *foundry_cli_options_get_filename       (FoundryCliOptions   *self,
                                                            const char          *key);
FOUNDRY_AVAILABLE_IN_ALL
void                foundry_cli_options_set_filename_array (FoundryCliOptions   *self,
                                                            const char          *key,
                                                            const char * const  *value);
FOUNDRY_AVAILABLE_IN_ALL
const char * const *foundry_cli_options_get_filename_array (FoundryCliOptions   *self,
                                                            const char          *key);
FOUNDRY_AVAILABLE_IN_ALL
void                foundry_cli_options_set_int            (FoundryCliOptions   *self,
                                                            const char          *key,
                                                            int                  value);
FOUNDRY_AVAILABLE_IN_ALL
gboolean            foundry_cli_options_get_int            (FoundryCliOptions   *self,
                                                            const char          *key,
                                                            int                 *value);
FOUNDRY_AVAILABLE_IN_ALL
void                foundry_cli_options_set_int64          (FoundryCliOptions   *self,
                                                            const char          *key,
                                                            gint64               value);
FOUNDRY_AVAILABLE_IN_ALL
gboolean            foundry_cli_options_get_int64          (FoundryCliOptions   *self,
                                                            const char          *key,
                                                            gint64              *value);
FOUNDRY_AVAILABLE_IN_ALL
void                foundry_cli_options_set_double         (FoundryCliOptions   *self,
                                                            const char          *key,
                                                            double               value);
FOUNDRY_AVAILABLE_IN_ALL
gboolean            foundry_cli_options_get_double         (FoundryCliOptions   *self,
                                                            const char          *key,
                                                            double              *value);
FOUNDRY_AVAILABLE_IN_ALL
void                foundry_cli_options_set_boolean        (FoundryCliOptions   *self,
                                                            const char          *key,
                                                            gboolean             value);
FOUNDRY_AVAILABLE_IN_ALL
gboolean            foundry_cli_options_get_boolean        (FoundryCliOptions   *self,
                                                            const char          *key,
                                                            gboolean            *value);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (FoundryCliOptions, foundry_cli_options_unref)

G_END_DECLS
