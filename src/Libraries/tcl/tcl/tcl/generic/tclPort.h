/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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
#ifndef _TCLPORT
#define _TCLPORT

#ifdef HAVE_TCL_CONFIG_H
#include "tclConfig.h"
#endif
#if defined(_WIN32)
#   include "tclWinPort.h"
#endif
#include "tcl.h"
#if !defined(_WIN32)
#   include "tclUnixPort.h"
#endif

#if defined(__CYGWIN__)
#   define USE_PUTENV 1
#   define USE_PUTENV_FOR_UNSET 1
/* On Cygwin, the environment is imported from the Cygwin DLL. */
    DLLIMPORT extern char **__cygwin_environ;
    DLLIMPORT extern int cygwin_conv_to_win32_path(const char *, char *);
#   define environ __cygwin_environ
#   define timezone _timezone
#endif

#if !defined(LLONG_MIN)
#   ifdef TCL_WIDE_INT_IS_LONG
#      define LLONG_MIN LONG_MIN
#   else
#      ifdef LLONG_BIT
#         define LLONG_MIN ((Tcl_WideInt)(Tcl_LongAsWide(1)<<(LLONG_BIT-1)))
#      else
/* Assume we're on a system with a 64-bit 'long long' type */
#         define LLONG_MIN ((Tcl_WideInt)(Tcl_LongAsWide(1)<<63))
#      endif
#   endif
/* Assume that if LLONG_MIN is undefined, then so is LLONG_MAX */
#   define LLONG_MAX (~LLONG_MIN)
#endif


#endif /* _TCLPORT */
