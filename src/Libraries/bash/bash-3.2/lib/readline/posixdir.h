/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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

/* This file should be included instead of <dirent.h> or <sys/dir.h>. */

#if !defined (_POSIXDIR_H_)
#define _POSIXDIR_H_

#if defined (HAVE_DIRENT_H)
#  include <dirent.h>
#  if defined (HAVE_STRUCT_DIRENT_D_NAMLEN)
#    define D_NAMLEN(d)	((d)->d_namlen)
#  else
#    define D_NAMLEN(d)   (strlen ((d)->d_name))
#  endif /* !HAVE_STRUCT_DIRENT_D_NAMLEN */
#else
#  if defined (HAVE_SYS_NDIR_H)
#    include <sys/ndir.h>
#  endif
#  if defined (HAVE_SYS_DIR_H)
#    include <sys/dir.h>
#  endif
#  if defined (HAVE_NDIR_H)
#    include <ndir.h>
#  endif
#  if !defined (dirent)
#    define dirent direct
#  endif /* !dirent */
#  define D_NAMLEN(d)   ((d)->d_namlen)
#endif /* !HAVE_DIRENT_H */

#if defined (HAVE_STRUCT_DIRENT_D_INO) && !defined (HAVE_STRUCT_DIRENT_D_FILENO)
#  define d_fileno d_ino
#endif

#if defined (_POSIX_SOURCE) && (!defined (HAVE_STRUCT_DIRENT_D_INO) || defined (BROKEN_DIRENT_D_INO))
/* Posix does not require that the d_ino field be present, and some
   systems do not provide it. */
#  define REAL_DIR_ENTRY(dp) 1
#else
#  define REAL_DIR_ENTRY(dp) (dp->d_ino != 0)
#endif /* _POSIX_SOURCE */

#endif /* !_POSIXDIR_H_ */
