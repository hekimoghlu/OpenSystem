/* foundry-command-line-private.h
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

#include "foundry-command-line.h"

G_BEGIN_DECLS

#define FOUNDRY_COMMAND_LINE_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST(klass, FOUNDRY_TYPE_COMMAND_LINE, FoundryCommandLineClass))
#define FOUNDRY_COMMAND_LINE_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS(obj, FOUNDRY_TYPE_COMMAND_LINE, FoundryCommandLineClass))

struct _FoundryCommandLine
{
  GObject parent_instance;
};

struct _FoundryCommandLineClass
{
  GObjectClass parent_class;

  char            *(*get_directory)   (FoundryCommandLine *self);
  DexFuture       *(*open)            (FoundryCommandLine *self,
                                       int                 fd_number);
  const char      *(*getenv)          (FoundryCommandLine *self,
                                       const char         *name);
  char           **(*get_environ)     (FoundryCommandLine *self);
  gboolean         (*isatty)          (FoundryCommandLine *self);
  void             (*print)           (FoundryCommandLine *self,
                                       const char         *message);
  void             (*printerr)        (FoundryCommandLine *self,
                                       const char         *message);
  DexFuture       *(*run)             (FoundryCommandLine *self,
                                       const char * const *argv);
  int              (*get_stdin)       (FoundryCommandLine *self);
  int              (*get_stdout)      (FoundryCommandLine *self);
  int              (*get_stderr)      (FoundryCommandLine *self);
  DexCancellable  *(*dup_cancellable) (FoundryCommandLine *self);
};

DexFuture *foundry_command_line_open               (FoundryCommandLine *self,
                                                    int                 fd_number)
  G_GNUC_WARN_UNUSED_RESULT;
void       foundry_command_line_help               (FoundryCommandLine *self);
void       foundry_command_line_clear_line         (FoundryCommandLine *self);

G_END_DECLS
