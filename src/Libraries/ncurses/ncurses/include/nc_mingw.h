/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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
 * Author: Thomas Dickey, 2008-on                                           * 
 *                                                                          *
 ****************************************************************************/

/* $Id: nc_mingw.h,v 1.3 2010/09/25 22:16:12 juergen Exp $ */

#ifndef NC_MINGW_H
#define NC_MINGW_H 1

#ifdef WINVER
#  if WINVER < 0x0501
#    error WINVER must at least be 0x0501
#  endif  
#else
#  define WINVER 0x0501
#endif
#include <windows.h>

#undef sleep
#define sleep(n) Sleep((n) * 1000)

#undef gettimeofday
#define gettimeofday(tv,tz) _nc_gettimeofday(tv,tz)

#include <sys/time.h>	/* for struct timeval */

extern int _nc_gettimeofday(struct timeval *, void *);

#undef HAVE_GETTIMEOFDAY
#define HAVE_GETTIMEOFDAY 1

#define SIGHUP  1
#define SIGKILL 9
#define getlogin() "username"

#undef wcwidth
#define wcwidth(ucs) _nc_wcwidth(ucs)
extern int _nc_wcwidth(wchar_t);

#endif /* NC_MINGW_H */
