/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 19, 2023.
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
#include <config.h>

#ifndef HAVE_HSTRERROR

#if (defined(SunOS) && (SunOS >= 50))
#define hstrerror broken_proto
#endif
#include "roken.h"
#if (defined(SunOS) && (SunOS >= 50))
#undef hstrerror
#endif

#if !(defined(HAVE_H_ERRLIST) && defined(HAVE_H_NERR))
static const char *const h_errlist[] = {
    "Resolver Error 0 (no error)",
    "Unknown host",		/* 1 HOST_NOT_FOUND */
    "Host name lookup failure",	/* 2 TRY_AGAIN */
    "Unknown server error",	/* 3 NO_RECOVERY */
    "No address associated with name", /* 4 NO_ADDRESS */
};

static
const
int h_nerr = { sizeof h_errlist / sizeof h_errlist[0] };
#else

#if !HAVE_DECL_H_ERRLIST
extern const char *h_errlist[];
extern int h_nerr;
#endif

#endif

ROKEN_LIB_FUNCTION const char * ROKEN_LIB_CALL
hstrerror(int herr)
{
    if (0 <= herr && herr < h_nerr)
	return h_errlist[herr];
    else if(herr == -17)
	return "unknown error";
    else
	return "Error number out of range (hstrerror)";
}

#endif
