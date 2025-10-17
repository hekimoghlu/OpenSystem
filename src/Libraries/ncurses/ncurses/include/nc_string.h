/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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
 *  Author: Thomas E. Dickey                        2012                    *
 ****************************************************************************/

#ifndef STRING_HACKS_H
#define STRING_HACKS_H 1

#include <ncurses_cfg.h>

/*
 * $Id: nc_string.h,v 1.4 2013/12/15 01:09:19 tom Exp $
 *
 * String-hacks.  Use these macros to stifle warnings on (presumably) correct
 * uses of strcat, strcpy and sprintf.
 *
 * By the way -
 * A fundamental limitation of the interfaces (and frequent issue in bug
 * reports using these functions) is that sizes are passed as unsigned values
 * (with associated sign-extension problems), limiting their effectiveness
 * when checking for buffer overflow.
 */

#ifdef __cplusplus
#define NCURSES_VOID		/* nothing */
#else
#define NCURSES_VOID (void)
#endif

#if USE_STRING_HACKS && HAVE_STRLCAT
#define _nc_STRCAT(d,s,n)	NCURSES_VOID strlcat((d),(s),NCURSES_CAST(size_t,n))
#else
#define _nc_STRCAT(d,s,n)	NCURSES_VOID strcat((d),(s))
#endif

#if USE_STRING_HACKS && HAVE_STRLCPY
#define _nc_STRCPY(d,s,n)	NCURSES_VOID strlcpy((d),(s),NCURSES_CAST(size_t,n))
#else
#define _nc_STRCPY(d,s,n)	NCURSES_VOID strcpy((d),(s))
#endif

#if USE_STRING_HACKS && HAVE_SNPRINTF
#define _nc_SPRINTF             NCURSES_VOID snprintf
#define _nc_SLIMIT(n)           NCURSES_CAST(size_t,n),
#else
#define _nc_SPRINTF             NCURSES_VOID sprintf
#define _nc_SLIMIT(n)		/* nothing */
#endif

#endif /* STRING_HACKS_H */
