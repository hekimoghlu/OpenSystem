/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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
#include "transformInt.h"
#include "dld.h"

/*
 *----------------------------------------------------------------------
 *
 * dlopen --
 *
 *	This function is an implementation of dlopen() using
 *	the dld library.
 *
 * Results:
 *	Returns the handle of the newly loaded library, or NULL on
 *	failure.
 *
 * Side effects:
 *	Loads the specified library into the process.
 *
 *----------------------------------------------------------------------
 */

static int returnCode = 0;

extern char *tclExecutableName;

VOID *dlopen(path, mode)
    CONST char *path;
    int mode;
{
    static int firstTime = 1;

    /*
     *  The dld package needs to know the pathname to the tcl binary.
     *  If that's not know, return an error.
     */

    returnCode = 0;
    if (firstTime) {
	if (tclExecutableName == NULL) {
	    return (VOID *) NULL;
	}
	returnCode = dld_init(tclExecutableName);
	if (returnCode != 0) {
	    return (VOID *) NULL;
	}
	firstTime = 0;
    }

    if ((path != NULL) && ((returnCode = dld_link(path)) != 0)) {
	return (VOID *) NULL;
    }

    return (VOID *) 1;
}

VOID *
dlsym(handle, symbol)
    VOID *handle;
    CONST char *symbol;
{
    return (VOID *) dld_get_func(symbol);
}

char *
dlerror()
{
    if (tclExecutableName == NULL) {
	return "don't know name of application binary file, so can't initialize dynamic loader";
    }
    return dld_strerror(returnCode);
}

int
dlclose(handle)
    VOID *handle;
{
    return 0;
}
