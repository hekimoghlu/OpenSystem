/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 16, 2021.
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
#ifndef	_STRMATCH_H
#define	_STRMATCH_H	1

#include <config.h>

#include "stdc.h"

/* We #undef these before defining them because some losing systems
   (HP-UX A.08.07 for example) define these in <unistd.h>.  */
#undef  FNM_PATHNAME
#undef  FNM_NOESCAPE
#undef  FNM_PERIOD

/* Bits set in the FLAGS argument to `strmatch'.  */

/* standard flags are like fnmatch(3). */
#define	FNM_PATHNAME	(1 << 0) /* No wildcard can ever match `/'.  */
#define	FNM_NOESCAPE	(1 << 1) /* Backslashes don't quote special chars.  */
#define	FNM_PERIOD	(1 << 2) /* Leading `.' is matched only explicitly.  */

/* extended flags not available in most libc fnmatch versions, but we undef
   them to avoid any possible warnings. */
#undef FNM_LEADING_DIR
#undef FNM_CASEFOLD
#undef FNM_EXTMATCH

#define FNM_LEADING_DIR	(1 << 3) /* Ignore `/...' after a match. */
#define FNM_CASEFOLD	(1 << 4) /* Compare without regard to case. */
#define FNM_EXTMATCH	(1 << 5) /* Use ksh-like extended matching. */

/* Value returned by `strmatch' if STRING does not match PATTERN.  */
#undef FNM_NOMATCH

#define	FNM_NOMATCH	1

/* Match STRING against the filename pattern PATTERN,
   returning zero if it matches, FNM_NOMATCH if not.  */
extern int strmatch __P((char *, char *, int));

#if HANDLE_MULTIBYTE
extern int wcsmatch __P((wchar_t *, wchar_t *, int));
#endif

#endif /* _STRMATCH_H */
