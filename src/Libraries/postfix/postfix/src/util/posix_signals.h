/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 23, 2024.
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

#ifndef _POSIX_SIGNALS_H_INCLUDED_
#define _POSIX_SIGNALS_H_INCLUDED_
/*++
/* NAME
/*	posix_signals 3h
/* SUMMARY
/*	POSIX signal handling compatibility
/* SYNOPSIS
/*	#include <posix_signals.h>
/* DESCRIPTION
/* .nf

 /*
  * Compatibility interface.
  */

#ifdef MISSING_SIGSET_T

typedef int sigset_t;

enum {
    SIG_BLOCK,
    SIG_UNBLOCK,
    SIG_SETMASK
};

extern int sigemptyset(sigset_t *);
extern int sigaddset(sigset_t *, int);
extern int sigprocmask(int, sigset_t *, sigset_t *);

#endif

#ifdef MISSING_SIGACTION

struct sigaction {
    void    (*sa_handler) ();
    sigset_t sa_mask;
    int     sa_flags;
};

 /* Possible values for sa_flags.  Or them to set multiple.  */
enum {
    SA_RESTART,
    SA_NOCLDSTOP = 4			/* drop the = 4.  */
};

extern int sigaction(int, struct sigaction *, struct sigaction *);

#endif

/* AUTHOR(S)
/*	Pieter Schoenmakers
/*	Eindhoven University of Technology
/*	P.O. Box 513
/*	5600 MB Eindhoven
/*	The Netherlands
/*--*/

#endif
