/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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

#if !defined (_UNWIND_PROT_H)
#define _UNWIND_PROT_H

/* Run a function without interrupts. */
extern void begin_unwind_frame __P((char *));
extern void discard_unwind_frame __P((char *));
extern void run_unwind_frame __P((char *));
extern void add_unwind_protect (); /* Not portable to arbitrary C99 hosts.  */
extern void remove_unwind_protect __P((void));
extern void run_unwind_protects __P((void));
extern void clear_unwind_protect_list __P((int));
extern void uwp_init __P((void));

/* Define for people who like their code to look a certain way. */
#define end_unwind_frame()

/* How to protect a variable.  */
#define unwind_protect_var(X) unwind_protect_mem ((char *)&(X), sizeof (X))
extern void unwind_protect_mem __P((char *, int));

/* Backwards compatibility */
#define unwind_protect_int	unwind_protect_var
#define unwind_protect_short	unwind_protect_var
#define unwind_protect_string	unwind_protect_var
#define unwind_protect_pointer	unwind_protect_var
#define unwind_protect_jmp_buf	unwind_protect_var

#endif /* _UNWIND_PROT_H */
