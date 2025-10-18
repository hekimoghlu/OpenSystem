/* egg-line.h
 *
 * Copyright 2009 Christian Hergert <chergert@redhat.com>
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

G_BEGIN_DECLS

typedef struct _EggLine EggLine;

typedef enum
{
	EGG_LINE_STATUS_OK,
	EGG_LINE_STATUS_BAD_ARGS,
	EGG_LINE_STATUS_FAILURE,
} EggLineStatus;

typedef struct _EggLineCommand EggLineCommand;

typedef EggLineCommand *(*EggLineGenerator) (EggLine          *line,
                                             int              *argc,
                                             char           ***argv);
typedef EggLineStatus   (*EggLineCallback)  (EggLine          *line,
                                             EggLineCommand   *command,
                                             int               argc,
                                             char            **argv,
                                             GError          **error);
typedef void            (*EggLineMissing)   (EggLine          *line,
                                             const char       *text);

struct _EggLineCommand
{
	const char        *name;
	EggLineGenerator   generator;
	EggLineCallback    callback;
	const char        *help;
	const char        *usage;
	gpointer           user_data;
};

EggLine        *egg_line_new                 (void);
EggLine        *egg_line_ref                 (EggLine                *self);
void            egg_line_unref               (EggLine                *self);
EggLineCommand *egg_line_resolve             (EggLine                *self,
                                              const char             *text,
                                              int                    *argc,
                                              char                 ***argv);
void            egg_line_set_missing_handler (EggLine                *self,
                                              EggLineMissing          missing);
void            egg_line_execute             (EggLine                *self,
                                              const char             *text);
void            egg_line_run                 (EggLine                *self);
void            egg_line_quit                (EggLine                *self);
void            egg_line_set_commands        (EggLine                *self,
                                              const EggLineCommand   *entries);
void            egg_line_set_prompt          (EggLine                *line,
                                              const char             *prompt);
void            egg_line_show_help           (EggLine                *line,
                                              const EggLineCommand   *entries);
void            egg_line_show_usage          (EggLine                *line,
                                              const EggLineCommand   *entry);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (EggLine, egg_line_unref)

G_END_DECLS

