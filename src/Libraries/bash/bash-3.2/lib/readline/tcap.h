/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 22, 2024.
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
/* Copyright (C) 1996 Free Software Foundation, Inc.

   This file contains the Readline Library (the Library), a set of
   routines for providing Emacs style line input to programs that ask
   for it.

   The Library is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   The Library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   The GNU General Public License is often shipped with GNU software, and
   is generally kept in a file called COPYING or LICENSE.  If you do not
   have a copy of the license, write to the Free Software Foundation,
   59 Temple Place, Suite 330, Boston, MA 02111 USA. */

#if !defined (_RLTCAP_H_)
#define _RLTCAP_H_

#if defined (HAVE_CONFIG_H)
#  include "config.h"
#endif

#if defined (HAVE_TERMCAP_H)
#  if defined (__linux__) && !defined (SPEED_T_IN_SYS_TYPES)
#    include "rltty.h"
#  endif
#  include <termcap.h>
#else

/* On Solaris2, sys/types.h #includes sys/reg.h, which #defines PC.
   Unfortunately, PC is a global variable used by the termcap library. */
#ifdef PC
#  undef PC
#endif

extern char PC;
extern char *UP, *BC;

extern short ospeed;

extern int tgetent ();
extern int tgetflag ();
extern int tgetnum ();
extern char *tgetstr ();

extern int tputs ();

extern char *tgoto ();

#endif /* HAVE_TERMCAP_H */

#endif /* !_RLTCAP_H_ */
