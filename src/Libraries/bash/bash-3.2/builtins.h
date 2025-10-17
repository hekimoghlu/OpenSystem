/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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
/* Copyright (C) 1987,1991 Free Software Foundation, Inc.

   This file is part of GNU Bash, the Bourne Again SHell.

   Bash is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   Bash is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
   License for more details.

   You should have received a copy of the GNU General Public License
   along with Bash; see the file COPYING.  If not, write to the Free
   Software Foundation, 59 Temple Place, Suite 330, Boston, MA 02111 USA. */

#include "config.h"

#if defined (HAVE_UNISTD_H)
#  ifdef _MINIX
#    include <sys/types.h>
#  endif
#  include <unistd.h>
#endif

#include "command.h"
#include "general.h"

#if defined (ALIAS)
#include "alias.h"
#endif

/* Flags describing various things about a builtin. */
#define BUILTIN_ENABLED 0x1	/* This builtin is enabled. */
#define BUILTIN_DELETED 0x2	/* This has been deleted with enable -d. */
#define STATIC_BUILTIN  0x4	/* This builtin is not dynamically loaded. */
#define SPECIAL_BUILTIN 0x8	/* This is a Posix `special' builtin. */
#define ASSIGNMENT_BUILTIN 0x10	/* This builtin takes assignment statements. */

#define BASE_INDENT	4

/* The thing that we build the array of builtins out of. */
struct builtin {
  char *name;			/* The name that the user types. */
  sh_builtin_func_t *function;	/* The address of the invoked function. */
  int flags;			/* One of the #defines above. */
  char * const *long_doc;	/* NULL terminated array of strings. */
  const char *short_doc;	/* Short version of documenation. */
  char *handle;			/* for future use */
};

/* Found in builtins.c, created by builtins/mkbuiltins. */
extern int num_shell_builtins;	/* Number of shell builtins. */
extern struct builtin static_shell_builtins[];
extern struct builtin *shell_builtins;
extern struct builtin *current_builtin;
