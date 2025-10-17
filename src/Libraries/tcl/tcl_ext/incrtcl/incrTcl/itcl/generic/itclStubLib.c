/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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
 * We need to ensure that we use the stub macros so that this file contains
 * no references to any of the stub functions.  This will make it possible
 * to build an extension that references Tcl_InitStubs but doesn't end up
 * including the rest of the stub functions.
 */

#ifndef USE_TCL_STUBS
#define USE_TCL_STUBS
#endif
#undef USE_TCL_STUB_PROCS

/*
 * This ensures that the Itcl_InitStubs has a prototype in
 * itcl.h and is not the macro that turns it into Tcl_PkgRequire
 */

#ifndef USE_ITCL_STUBS
#define USE_ITCL_STUBS
#endif

#include "itclInt.h"

ItclStubs *itclStubsPtr;
ItclIntStubs *itclIntStubsPtr;

/*
 *----------------------------------------------------------------------
 *
 * Itcl_InitStubs --
 *
 *	Tries to initialize the stub table pointers and ensures that
 *	the correct version of Itcl is loaded.
 *
 * Results:
 *	The actual version of Itcl that satisfies the request, or
 *	NULL to indicate that an error occurred.
 *
 * Side effects:
 *	Sets the stub table pointers.
 *
 *----------------------------------------------------------------------
 */

#ifdef Itcl_InitStubs
#undef Itcl_InitStubs
#endif

CONST char *
Itcl_InitStubs (interp, version, exact)
    Tcl_Interp *interp;
    CONST char *version;
    int exact;
{
    CONST char *actualVersion;
    
    actualVersion = Tcl_PkgRequireEx(interp, "Itcl", (CONST84 char *)version, exact,
        (ClientData *) &itclStubsPtr);

    if (actualVersion == NULL) {
	itclStubsPtr = NULL;
	return NULL;
    }

    if (itclStubsPtr->hooks) {
	itclIntStubsPtr = itclStubsPtr->hooks->itclIntStubs;
    } else {
	itclIntStubsPtr = NULL;
    }
    
    return actualVersion;
}
