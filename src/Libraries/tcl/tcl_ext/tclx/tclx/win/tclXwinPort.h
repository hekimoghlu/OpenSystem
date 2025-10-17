/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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
#ifndef TCLXWINPORT_H
#define TCLXWINPORT_H

#include "tclWinPort.h"

#include <direct.h>
#include <assert.h>

/*
 * Types needed for fstat, but are not directly supported (we emulate).  If
 * defined before tclWinPort.h is include, it will define the access macros.
 */
#define S_IFIFO  _S_IFIFO               /* pipe */
#define S_IFSOCK 0140000                /* socket */

/*
 * OS feature definitons.
 */
#ifndef NO_CATGETS
#   define NO_CATGETS
#endif
#ifndef NO_FCHMOD
#   define NO_FCHMOD
#endif
#ifndef NO_FCHOWN
#   define NO_FCHOWN
#endif
#ifndef NO_FSYNC
#   define NO_FSYNC
#endif
#ifndef NO_RANDOM
#   define NO_RANDOM  /* uses compat */
#endif
#ifndef NO_SIGACTION
#   define NO_SIGACTION
#endif
#ifndef NO_TRUNCATE
#   define NO_TRUNCATE    /* FIX: Are we sure there is no way to truncate???*/
#endif
#ifndef RETSIGTYPE
#   define RETSIGTYPE void
#endif

#include <math.h>
#include <limits.h>

#ifndef MAXDOUBLE
#    define MAXDOUBLE HUGE_VAL
#endif

/*
 * No restartable signals in WIN32.
 */
#ifndef NO_SIG_RESTART
#   define NO_SIG_RESTART
#endif

/*
 * Define a macro to call wait pid.  We don't use Tcl_WaitPid on Unix because
 * it delays signals.
 */
#define TCLX_WAITPID(pid, status, options) \
	Tcl_WaitPid((Tcl_Pid)pid, status, options)

#define bcopy(from, to, length)    memmove((to), (from), (length))

/*
 * Compaibility functions.
 */
extern long	random(void);

extern void	srandom(unsigned int x);

extern int	getopt(int nargc, char * const *nargv, const char *ostr);

#endif


