/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 28, 2024.
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
/*
 * $FreeBSD: src/usr.sbin/cron/cron/compat.h,v 1.5 1999/08/28 01:15:49 peter Exp $
 */

#ifndef __P
# ifdef __STDC__
#  define __P(x) x
# else
#  define __P(x) ()
#  define const
# endif
#endif

#if defined(UNIXPC) || defined(unixpc)
# define UNIXPC 1
# define ATT 1
#endif

#if defined(hpux) || defined(_hpux) || defined(__hpux)
# define HPUX 1
# define seteuid(e) setresuid(-1,e,-1)
# define setreuid(r,e)	setresuid(r,e,-1)
#endif

#if defined(_IBMR2)
# define AIX 1
#endif

#if defined(__convex__)
# define CONVEX 1
#endif

#if defined(sgi) || defined(_sgi) || defined(__sgi)
# define IRIX 1
/* IRIX 4 hdrs are broken: one cannot #include both <stdio.h>
 * and <stdlib.h> because they disagree on system(), perror().
 * Therefore we must zap the "const" keyword BEFORE including
 * either of them.
 */
# define const
#endif

#if defined(_UNICOS)
# define UNICOS 1
#endif

#ifndef POSIX
# if (BSD >= 199103) || defined(__linux) || defined(ultrix) || defined(AIX) ||\
	defined(HPUX) || defined(CONVEX) || defined(IRIX)
#  define POSIX
# endif
#endif

#ifndef BSD
# if defined(ultrix)
#  define BSD 198902
# endif
#endif

/*****************************************************************/

#if !defined(BSD) && !defined(HPUX) && !defined(CONVEX) && !defined(__linux)
# define NEED_VFORK
#endif

#if (!defined(BSD) || (BSD < 198902)) && !defined(__linux) && \
	!defined(IRIX) && !defined(NeXT) && !defined(HPUX)
# define NEED_STRCASECMP
#endif

#if (!defined(BSD) || (BSD < 198911)) && !defined(__linux) &&\
	!defined(IRIX) && !defined(UNICOS) && !defined(HPUX)
# define NEED_STRDUP
#endif

#if (!defined(BSD) || (BSD < 198911)) && !defined(POSIX) && !defined(NeXT)
# define NEED_STRERROR
#endif

#if defined(HPUX) || defined(AIX) || defined(UNIXPC)
# define NEED_FLOCK
#endif

#ifndef POSIX
# define NEED_SETSID
#endif

#if (defined(POSIX) && !defined(BSD)) && !defined(__linux)
# define NEED_GETDTABLESIZE
#endif

#ifdef POSIX
#include <unistd.h>
#ifdef _POSIX_SAVED_IDS
# define HAVE_SAVED_UIDS
#endif
#endif

#if !defined(ATT) && !defined(__linux) && !defined(IRIX) && !defined(UNICOS)
# define USE_SIGCHLD
#endif

#if !defined(AIX) && !defined(UNICOS)
# define SYS_TIME_H 1
#else
# define SYS_TIME_H 0
#endif

#if defined(BSD) && !defined(POSIX)
# define USE_UTIMES
#endif

#if defined(AIX) || defined(HPUX) || defined(IRIX)
# define NEED_SETENV
#endif

#if !defined(UNICOS) && !defined(UNIXPC)
# define HAS_FCHOWN
#endif

#if !defined(UNICOS) && !defined(UNIXPC)
# define HAS_FCHMOD
#endif
