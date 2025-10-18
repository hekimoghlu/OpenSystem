/* foundry-process-launcher.h
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

#if !defined (FOUNDRY_INSIDE) && !defined (FOUNDRY_COMPILATION)
# error "Only <foundry.h> can be included directly."
#endif

#include <gio/gio.h>

#include "foundry-unix-fd-map.h"

G_BEGIN_DECLS

#define FOUNDRY_TYPE_PROCESS_LAUNCHER (foundry_process_launcher_get_type())

FOUNDRY_AVAILABLE_IN_ALL
G_DECLARE_FINAL_TYPE (FoundryProcessLauncher, foundry_process_launcher, FOUNDRY, PROCESS_LAUNCHER, GObject)

/**
 * FoundryProcessLauncherShell:
 * @FOUNDRY_PROCESS_LAUNCHER_SHELL_DEFAULT: A basic shell with no user scripts
 * @FOUNDRY_PROCESS_LAUNCHER_SHELL_LOGIN: A user login shell similar to `bash -l`
 * @FOUNDRY_PROCESS_LAUNCHER_SHELL_INTERACTIVE: A user interactive shell similar to `bash -i`
 *
 * Describes the type of shell to be used within the context.
 */
typedef enum _FoundryProcessLauncherShell
{
  FOUNDRY_PROCESS_LAUNCHER_SHELL_DEFAULT     = 0,
  FOUNDRY_PROCESS_LAUNCHER_SHELL_LOGIN       = 1,
  FOUNDRY_PROCESS_LAUNCHER_SHELL_INTERACTIVE = 2,
} FoundryProcessLauncherShell;

/**
 * FoundryProcessLauncherHandler:
 *
 * Returns: %TRUE if successful; otherwise %FALSE and @error must be set.
 */
typedef gboolean (*FoundryProcessLauncherHandler) (FoundryProcessLauncher  *process_launcher,
                                                   const char * const      *argv,
                                                   const char * const      *env,
                                                   const char              *cwd,
                                                   FoundryUnixFDMap        *unix_fd_map,
                                                   gpointer                 user_data,
                                                   GError                 **error);

FOUNDRY_AVAILABLE_IN_ALL
FoundryProcessLauncher *foundry_process_launcher_new                     (void);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_push                    (FoundryProcessLauncher         *self,
                                                                          FoundryProcessLauncherHandler   handler,
                                                                          gpointer                        handler_data,
                                                                          GDestroyNotify                  handler_data_destroy);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_push_at_base            (FoundryProcessLauncher         *self,
                                                                          FoundryProcessLauncherHandler   handler,
                                                                          gpointer                        handler_data,
                                                                          GDestroyNotify                  handler_data_destroy);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_push_error              (FoundryProcessLauncher         *self,
                                                                          GError                         *error);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_push_host               (FoundryProcessLauncher         *self);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_push_shell              (FoundryProcessLauncher         *self,
                                                                          FoundryProcessLauncherShell     shell);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_push_user_shell         (FoundryProcessLauncher         *self,
                                                                          FoundryProcessLauncherShell     shell);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_push_expansion          (FoundryProcessLauncher         *self,
                                                                          const char * const             *environ);
FOUNDRY_AVAILABLE_IN_ALL
const char * const     *foundry_process_launcher_get_argv                (FoundryProcessLauncher         *self);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_set_argv                (FoundryProcessLauncher         *self,
                                                                          const char * const             *argv);
FOUNDRY_AVAILABLE_IN_ALL
const char * const     *foundry_process_launcher_get_environ             (FoundryProcessLauncher         *self);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_set_environ             (FoundryProcessLauncher         *self,
                                                                          const char * const             *environ);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_add_environ             (FoundryProcessLauncher         *self,
                                                                          const char * const             *environ);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_add_minimal_environment (FoundryProcessLauncher         *self);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_environ_to_argv         (FoundryProcessLauncher         *self);
FOUNDRY_AVAILABLE_IN_ALL
const char             *foundry_process_launcher_get_cwd                 (FoundryProcessLauncher         *self);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_set_cwd                 (FoundryProcessLauncher         *self,
                                                                          const char                     *cwd);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_set_pty_fd              (FoundryProcessLauncher         *self,
                                                                          int                             consumer_fd);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_take_fd                 (FoundryProcessLauncher         *self,
                                                                          int                             source_fd,
                                                                          int                             dest_fd);
FOUNDRY_AVAILABLE_IN_ALL
gboolean                foundry_process_launcher_merge_unix_fd_map       (FoundryProcessLauncher         *self,
                                                                          FoundryUnixFDMap               *unix_fd_map,
                                                                          GError                        **error);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_prepend_argv            (FoundryProcessLauncher         *self,
                                                                          const char                     *arg);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_prepend_args            (FoundryProcessLauncher         *self,
                                                                          const char * const             *args);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_append_argv             (FoundryProcessLauncher         *self,
                                                                          const char                     *arg);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_append_args             (FoundryProcessLauncher         *self,
                                                                          const char * const             *args);
FOUNDRY_AVAILABLE_IN_ALL
gboolean                foundry_process_launcher_append_args_parsed      (FoundryProcessLauncher         *self,
                                                                          const char                     *args,
                                                                          GError                        **error);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_append_formatted        (FoundryProcessLauncher         *self,
                                                                          const char                     *format,
                                                                          ...) G_GNUC_PRINTF (2, 3);
FOUNDRY_AVAILABLE_IN_ALL
const char             *foundry_process_launcher_getenv                  (FoundryProcessLauncher         *self,
                                                                          const char                     *key);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_setenv                  (FoundryProcessLauncher         *self,
                                                                          const char                     *key,
                                                                          const char                     *value);
FOUNDRY_AVAILABLE_IN_ALL
void                    foundry_process_launcher_unsetenv                (FoundryProcessLauncher         *self,
                                                                          const char                     *key);
FOUNDRY_AVAILABLE_IN_ALL
GIOStream              *foundry_process_launcher_create_stdio_stream     (FoundryProcessLauncher         *self,
                                                                          GError                        **error);
FOUNDRY_AVAILABLE_IN_ALL
GSubprocess            *foundry_process_launcher_spawn                   (FoundryProcessLauncher         *self,
                                                                          GError                        **error);
FOUNDRY_AVAILABLE_IN_ALL
GSubprocess            *foundry_process_launcher_spawn_with_flags        (FoundryProcessLauncher         *self,
                                                                          GSubprocessFlags                flags,
                                                                          GError                        **error);
FOUNDRY_AVAILABLE_IN_ALL
int                     foundry_pty_create_producer                      (int                             pty_consumer_fd,
                                                                          gboolean                        blocking,
                                                                          GError                        **error);

G_END_DECLS
