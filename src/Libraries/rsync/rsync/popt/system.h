/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 12, 2022.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>

#if HAVE_MCHECK_H 
#include <mcheck.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if HAVE_UNISTD_H
#include <unistd.h>
#endif

#if !defined(__GNUC__) || defined(APPLE)
/* Apparently the OS X port of gcc gags on __attribute__.
 *
 * <http://www.opensource.apple.com/bugs/X/gcc/2512150.html> */
#define __attribute__(x) 

#endif

#ifdef __NeXT
/* access macros are not declared in non posix mode in unistd.h -
 don't try to use posix on NeXTstep 3.3 ! */
#include <libc.h>
#endif

#if defined(__LCLINT__)
/*@-declundef -incondefs -redecl@*/ /* LCL: missing annotation */
/*@only@*/ void * alloca (size_t __size)
	/*@ensures MaxSet(result) == (__size - 1) @*/
	/*@*/;
/*@=declundef =incondefs =redecl@*/
#endif

/* AIX requires this to be the first thing in the file.  */ 
#ifndef __GNUC__
# if HAVE_ALLOCA_H
#  include <alloca.h>
# else
#  ifdef _AIX
#pragma alloca
#  else
#   if HAVE_ALLOCA
#    ifndef alloca /* predefined by HP cc +Olibcalls */
char *alloca ();
#    endif
#   else
#    ifdef alloca
#     undef alloca
#    endif
#    define alloca(sz) malloc(sz) /* Kludge this for now */
#   endif
#  endif
# endif
#elif defined(__GNUC__) && defined(__STRICT_ANSI__)
#define alloca __builtin_alloca
#endif

/*@-redecl -redef@*/
/*@mayexit@*/ /*@only@*/ char * xstrdup (const char *str)
	/*@*/;
/*@=redecl =redef@*/

#if HAVE_MCHECK_H && defined(__GNUC__)
#define	vmefail()	(fprintf(stderr, "virtual memory exhausted.\n"), exit(EXIT_FAILURE), NULL)
#define xstrdup(_str)   (strcpy((malloc(strlen(_str)+1) ? : vmefail()), (_str)))
#else
#define	xstrdup(_str)	strdup(_str)
#endif  /* HAVE_MCHECK_H && defined(__GNUC__) */

#define UNUSED(x) x __attribute__((__unused__))

#include "popt.h"
