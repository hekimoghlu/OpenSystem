/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 20, 2021.
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

#if !defined (_SIGLIST_H_)
#define _SIGLIST_H_

#if !defined (SYS_SIGLIST_DECLARED) && !defined (HAVE_STRSIGNAL)

#if defined (HAVE_UNDER_SYS_SIGLIST) && !defined (HAVE_SYS_SIGLIST) && !defined (sys_siglist)
#  define sys_siglist _sys_siglist
#endif /* HAVE_UNDER_SYS_SIGLIST && !HAVE_SYS_SIGLIST && !sys_siglist */

#if !defined (sys_siglist)
extern char *sys_siglist[];
#endif /* !sys_siglist */

#endif /* !SYS_SIGLIST_DECLARED && !HAVE_STRSIGNAL */

#if !defined (strsignal) && !defined (HAVE_STRSIGNAL)
#  define strsignal(sig) (char *)sys_siglist[sig]
#endif /* !strsignal && !HAVE_STRSIGNAL */

#if !defined (strsignal) && !HAVE_DECL_STRSIGNAL
extern char *strsignal __P((int));
#endif

#endif /* _SIGLIST_H */
