/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 17, 2022.
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
/****************************************************************************
 *  Author: Zeyd M. Ben-Halim <zmbenhal@netcom.com> 1992,1995               *
 *     and: Eric S. Raymond <esr@snark.thyrsus.com>                         *
 ****************************************************************************/

/*
 * $Id: SigAction.h,v 1.8 2005/08/06 20:05:32 tom Exp $
 *
 * This file exists to handle non-POSIX systems which don't have <unistd.h>,
 * and usually no sigaction() nor <termios.h>
 */

#ifndef _SIGACTION_H
#define _SIGACTION_H

#ifndef HAVE_SIGACTION
#define HAVE_SIGACTION 0
#endif

#ifndef HAVE_SIGVEC
#define HAVE_SIGVEC 0
#endif

#if HAVE_SIGACTION

#if !HAVE_TYPE_SIGACTION
typedef struct sigaction sigaction_t;
#endif

#else	/* !HAVE_SIGACTION */

#if HAVE_SIGVEC

#undef  SIG_BLOCK
#define SIG_BLOCK       00

#undef  SIG_UNBLOCK
#define SIG_UNBLOCK     01

#undef  SIG_SETMASK
#define SIG_SETMASK     02

 	/*
	 * <bsd/signal.h> is in the Linux 1.2.8 + gcc 2.7.0 configuration,
	 * and is useful for testing this header file.
	 */
#if HAVE_BSD_SIGNAL_H
#include <bsd/signal.h>
#endif

typedef struct sigvec sigaction_t;

#define sigset_t _nc_sigset_t
typedef unsigned long sigset_t;

#undef  sa_mask
#define sa_mask sv_mask
#undef  sa_handler
#define sa_handler sv_handler
#undef  sa_flags
#define sa_flags sv_flags

#undef  sigaction
#define sigaction   _nc_sigaction
#undef  sigprocmask
#define sigprocmask _nc_sigprocmask
#undef  sigemptyset
#define sigemptyset _nc_sigemptyset
#undef  sigsuspend
#define sigsuspend  _nc_sigsuspend
#undef  sigdelset
#define sigdelset   _nc_sigdelset
#undef  sigaddset
#define sigaddset   _nc_sigaddset

/* tty/lib_tstp.c is the only user */
#include <base/sigaction.c>

#endif /* HAVE_SIGVEC */
#endif /* HAVE_SIGACTION */
#endif /* !defined(_SIGACTION_H) */
