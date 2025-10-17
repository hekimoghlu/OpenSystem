/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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

#include "roken.h"

#ifndef HAVE___PROGNAME
extern const char *__progname;
#endif

#ifndef HAVE_SETPROGNAME

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
setprogname(const char *argv0)
{

#ifndef HAVE___PROGNAME

    const char *p;
    if(argv0 == NULL)
	return;
    p = strrchr(argv0, '/');

#ifdef BACKSLASH_PATH_DELIM
    {
        const char * pb;

        pb = strrchr((p != NULL)? p : argv0, '\\');
        if (pb != NULL)
            p = pb;
    }
#endif

    if(p == NULL)
	p = argv0;
    else
	p++;

#ifdef _WIN32
    {
        char * fn = strdup(p);
        char * ext;

        strlwr(fn);
        ext = strrchr(fn, '.');
        if (ext != NULL && !strcmp(ext, ".exe"))
            *ext = '\0';

        __progname = fn;
    }
#else

    __progname = p;

#endif

#endif  /* HAVE___PROGNAME */
}

#endif /* HAVE_SETPROGNAME */
