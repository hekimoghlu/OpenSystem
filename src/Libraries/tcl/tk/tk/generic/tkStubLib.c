/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 14, 2023.
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
 * We need to ensure that we use the stub macros so that this file contains no
 * references to any of the stub functions. This will make it possible to
 * build an extension that references Tk_InitStubs but doesn't end up
 * including the rest of the stub functions.
 */

#ifndef USE_TCL_STUBS
#define USE_TCL_STUBS
#endif
#undef USE_TCL_STUB_PROCS

#ifndef USE_TK_STUBS
#define USE_TK_STUBS
#endif
#undef USE_TK_STUB_PROCS

#include "tkInt.h"

#ifdef __WIN32__
#include "tkWinInt.h"
#endif

#ifdef MAC_OSX_TK
#include "tkMacOSXInt.h"
#endif

#if !(defined(__WIN32__) || defined(MAC_OSX_TK))
#include "tkUnixInt.h"
#endif

/* TODO: These ought to come in some other way */
#include "tkPlatDecls.h"
#include "tkIntXlibDecls.h"

TkStubs *tkStubsPtr = NULL;
TkPlatStubs *tkPlatStubsPtr = NULL;
TkIntStubs *tkIntStubsPtr = NULL;
TkIntPlatStubs *tkIntPlatStubsPtr = NULL;
TkIntXlibStubs *tkIntXlibStubsPtr = NULL;

/*
 * Use our own isdigit to avoid linking to libc on windows
 */

static int isDigit(const int c)
{
    return (c >= '0' && c <= '9');
}

/*
 *----------------------------------------------------------------------
 *
 * Tk_InitStubs --
 *
 *	Checks that the correct version of Tk is loaded and that it supports
 *	stubs. It then initialises the stub table pointers.
 *
 * Results:
 *	The actual version of Tk that satisfies the request, or NULL to
 *	indicate that an error occurred.
 *
 * Side effects:
 *	Sets the stub table pointers.
 *
 *----------------------------------------------------------------------
 */

#ifdef Tk_InitStubs
#undef Tk_InitStubs
#endif

CONST char *
Tk_InitStubs(
    Tcl_Interp *interp,
    CONST char *version,
    int exact)
{
    CONST char *actualVersion;
    TkStubs **stubsPtrPtr = &tkStubsPtr;	/* squelch warning */

    actualVersion = Tcl_PkgRequireEx(interp, "Tk", version, 0,
	    (ClientData *) stubsPtrPtr);
    if (!actualVersion) {
	return NULL;
    }
    if (exact) {
        CONST char *p = version;
        int count = 0;

        while (*p) {
            count += !isDigit(*p++);
        }
        if (count == 1) {
	    CONST char *q = actualVersion;

	    p = version;
	    while (*p && (*p == *q)) {
		p++; q++;
	    }
            if (*p) {
		/* Construct error message */
		Tcl_PkgRequireEx(interp, "Tk", version, 1, NULL);
                return NULL;

            }
        } else {
            actualVersion = Tcl_PkgRequireEx(interp, "Tk", version, 1, NULL);
            if (actualVersion == NULL) {
                return NULL;
            }
        }
    }

    if (!tkStubsPtr) {
	Tcl_SetResult(interp,
		"This implementation of Tk does not support stubs",
		TCL_STATIC);
	return NULL;
    }

    tkPlatStubsPtr = tkStubsPtr->hooks->tkPlatStubs;
    tkIntStubsPtr = tkStubsPtr->hooks->tkIntStubs;
    tkIntPlatStubsPtr = tkStubsPtr->hooks->tkIntPlatStubs;
    tkIntXlibStubsPtr = tkStubsPtr->hooks->tkIntXlibStubs;

    return actualVersion;
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
