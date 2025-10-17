/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 13, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
/* Copyright (C) 1993 Free Software Foundation, Inc.

   This file is part of GNU Bash, the Bourne Again SHell.

   Bash is free software; you can redistribute it and/or modify it under
   the terms of the GNU General Public License as published by the Free
   Software Foundation; either version 2, or (at your option) any later
   version.

   Bash is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   You should have received a copy of the GNU General Public License along
   with Bash; see the file COPYING.  If not, write to the Free Software
   Foundation, 59 Temple Place, Suite 330, Boston, MA 02111 USA. */

#if !defined (_FLAGS_H_)
#define _FLAGS_H_

#include "stdc.h"

/* Welcome to the world of Un*x, where everything is slightly backwards. */
#define FLAG_ON '-'
#define FLAG_OFF '+'

#define FLAG_ERROR -1
#define FLAG_UNKNOWN (int *)0

/* The thing that we build the array of flags out of. */
struct flags_alist {
  char name;
  int *value;
};

extern struct flags_alist shell_flags[];
extern char optflags[];

extern int
  mark_modified_vars, exit_immediately_on_error, disallow_filename_globbing,
  place_keywords_in_env, read_but_dont_execute,
  just_one_command, unbound_vars_is_error, echo_input_at_read,
  echo_command_at_execute, no_invisible_vars, noclobber,
  hashing_enabled, forced_interactive, privileged_mode,
  asynchronous_notification, interactive_comments, no_symbolic_links,
  function_trace_mode, error_trace_mode, pipefail_opt;

#if 0
extern int lexical_scoping;
#endif

#if defined (BRACE_EXPANSION)
extern int brace_expansion;
#endif

#if defined (BANG_HISTORY)
extern int history_expansion;
#endif /* BANG_HISTORY */

#if defined (RESTRICTED_SHELL)
extern int restricted;
extern int restricted_shell;
#endif /* RESTRICTED_SHELL */

extern int *find_flag __P((int));
extern int change_flag __P((int, int));
extern char *which_set_flags __P((void));
extern void reset_shell_flags __P((void));

extern void initialize_flags __P((void));

/* A macro for efficiency. */
#define change_flag_char(flag, on_or_off)  change_flag (flag, on_or_off)

#endif /* _FLAGS_H_ */
