/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 11, 2024.
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
 * Copyright 2008 Purdue Research Foundation, West Lafayette, Indiana
 * 47907.  All rights reserved.
 *
 * Written by Victor A. Abell
 *
 * This software is not subject to any license of the American Telephone
 * and Telegraph Company or the Regents of the University of California.
 *
 * Permission is granted to anyone to use this software for any purpose on
 * any computer system, and to alter it and redistribute it freely, subject
 * to the following restrictions:
 *
 * 1. Neither the authors nor Purdue University are responsible for any
 *    consequences of the use of this software.
 *
 * 2. The origin of this software must not be misrepresented, either by
 *    explicit claim or by omission.  Credit to the authors and Purdue
 *    University must appear in documentation and sources.
 *
 * 3. Altered versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 *
 * 4. This notice may not be removed or altered.
 */


#ifndef lint
static char copyright[] =
"@(#) Copyright 2008 Purdue Research Foundation.\nAll rights reserved.\n";
static char *rcsid = "$Id: util.c,v 1.1 2008/04/01 11:56:53 abe Exp $";
#endif

#if	defined(HAS_STRFTIME)
#include <time.h>
#endif	/* defined(HAS_STRFTIME) */


/*
 * util_strftime() -- utility function to call strftime(3) without header
 *		      file distractions
 */

int
util_strftime(fmtr, fmtl, fmt)
	char *fmtr;			/* format output receiver */
	int fmtl;			/* sizeof(*fmtr) */
	char *fmt;			/* format */
{

#if	defined(HAS_STRFTIME)
	struct tm *lt;
	time_t tm;

	tm = time((time_t *)NULL);
	lt = localtime(&tm);
	return(strftime(fmtr, fmtl, fmt, lt));
#else	/* !defined(HAS_STRFTIME) */
	return(0);
#endif	/* defined(HAS_STRFTIME) */

}
