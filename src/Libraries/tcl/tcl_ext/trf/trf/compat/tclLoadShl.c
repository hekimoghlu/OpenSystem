/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 17, 2023.
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
#include <dl.h>

/*
 * On some HP machines, dl.h defines EXTERN; remove that definition.
 */

#ifdef EXTERN
#   undef EXTERN
#endif

#include "transformInt.h"

#ifndef DYNAMIC_PATH
#    define DYNAMIC_PATH 0
#endif

VOID *dlopen(path, mode)
    CONST char *path;
#if defined(__hpux) && defined(__ia64)
    int mode;
#else
    unsigned int mode;
#endif
{
    int flags, length;

    if (path == (char *) NULL) {
	return (VOID *) PROG_HANDLE;
    }
    flags = ((mode & RTLD_NOW) ? BIND_IMMEDIATE : BIND_DEFERRED) |
	    DYNAMIC_PATH;
#ifdef BIND_VERBOSE
    length = strlen(path);
    if ((length > 2) && !(strcmp(path+length-3,".sl"))) {
	flags |= BIND_VERBOSE;
    }
#endif
    return (VOID *) shl_load(path, flags, 0L);
}

VOID *dlsym(handle, symbol)
    VOID *handle;
    CONST char *symbol;
{   VOID *address;

    if (shl_findsym((shl_t *)&handle, symbol,
	    (short) TYPE_UNDEFINED, &address) != 0) {
	address = NULL;
    }
    return address;
}

char *dlerror()
{
    return Tcl_ErrnoMsg(errno);
}

int dlclose(handle)
    VOID *handle;
{
    return shl_unload((shl_t) handle);
}
