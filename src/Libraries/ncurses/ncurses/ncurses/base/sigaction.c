/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 15, 2023.
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
 *     and: Thomas E. Dickey                        1996-2003               *
 ****************************************************************************/

/* This file provides sigaction() emulation using sigvec() */
/* Use only if this is non POSIX system */

MODULE_ID("$Id: sigaction.c,v 1.14 2003/12/07 01:06:52 tom Exp $")

static int
_nc_sigaction(int sig, sigaction_t * sigact, sigaction_t * osigact)
{
    return sigvec(sig, sigact, osigact);
}

static int
_nc_sigemptyset(sigset_t * mask)
{
    *mask = 0;
    return 0;
}

static int
_nc_sigprocmask(int mode, sigset_t * mask, sigset_t * omask)
{
    sigset_t current = sigsetmask(0);

    if (omask)
	*omask = current;

    if (mode == SIG_BLOCK)
	current |= *mask;
    else if (mode == SIG_UNBLOCK)
	current &= ~*mask;
    else if (mode == SIG_SETMASK)
	current = *mask;

    sigsetmask(current);
    return 0;
}

static int
_nc_sigaddset(sigset_t * mask, int sig)
{
    *mask |= sigmask(sig);
    return 0;
}

/* not used in lib_tstp.c */
#if 0
static int
_nc_sigsuspend(sigset_t * mask)
{
    return sigpause(*mask);
}

static int
_nc_sigdelset(sigset_t * mask, int sig)
{
    *mask &= ~sigmask(sig);
    return 0;
}

static int
_nc_sigismember(sigset_t * mask, int sig)
{
    return (*mask & sigmask(sig)) != 0;
}
#endif
