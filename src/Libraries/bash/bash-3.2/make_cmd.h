/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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
/* Copyright (C) 1993-2005 Free Software Foundation, Inc.

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

#if !defined (_MAKE_CMD_H_)
#define _MAKE_CMD_H_

#include "stdc.h"

extern void cmd_init __P((void));

extern WORD_DESC *alloc_word_desc __P((void));
extern WORD_DESC *make_bare_word __P((const char *));
extern WORD_DESC *make_word_flags __P((WORD_DESC *, const char *));
extern WORD_DESC *make_word __P((const char *));
extern WORD_DESC *make_word_from_token __P((int));

extern WORD_LIST *make_word_list __P((WORD_DESC *, WORD_LIST *));

#define add_string_to_list(s, l) make_word_list (make_word(s), (l))

extern COMMAND *make_command __P((enum command_type, SIMPLE_COM *));
extern COMMAND *command_connect __P((COMMAND *, COMMAND *, int));
extern COMMAND *make_for_command __P((WORD_DESC *, WORD_LIST *, COMMAND *, int));
extern COMMAND *make_group_command __P((COMMAND *));
extern COMMAND *make_case_command __P((WORD_DESC *, PATTERN_LIST *, int));
extern PATTERN_LIST *make_pattern_list __P((WORD_LIST *, COMMAND *));
extern COMMAND *make_if_command __P((COMMAND *, COMMAND *, COMMAND *));
extern COMMAND *make_while_command __P((COMMAND *, COMMAND *));
extern COMMAND *make_until_command __P((COMMAND *, COMMAND *));
extern COMMAND *make_bare_simple_command __P((void));
extern COMMAND *make_simple_command __P((ELEMENT, COMMAND *));
extern void make_here_document __P((REDIRECT *));
extern REDIRECT *make_redirection __P((int, enum r_instruction, REDIRECTEE));
extern COMMAND *make_function_def __P((WORD_DESC *, COMMAND *, int, int));
extern COMMAND *clean_simple_command __P((COMMAND *));

extern COMMAND *make_arith_command __P((WORD_LIST *));

extern COMMAND *make_select_command __P((WORD_DESC *, WORD_LIST *, COMMAND *, int));

#if defined (COND_COMMAND)
extern COND_COM *make_cond_node __P((int, WORD_DESC *, COND_COM *, COND_COM *));
extern COMMAND *make_cond_command __P((COND_COM *));
#endif

extern COMMAND *make_arith_for_command __P((WORD_LIST *, COMMAND *, int));

extern COMMAND *make_subshell_command __P((COMMAND *));

extern COMMAND *connect_async_list __P((COMMAND *, COMMAND *, int));

#endif /* !_MAKE_CMD_H */
