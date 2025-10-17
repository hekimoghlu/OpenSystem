/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 8, 2024.
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
#ifndef USE_TCL_STUBS
#  define USE_TCL_STUBS
#endif
#undef USE_TCL_STUB_PROCS

#include <tdom.h>

/*
 * Ensure that Tdom_InitStubs is built as an exported symbol.  The other stub
 * functions should be built as non-exported symbols.
 */

#undef TCL_STORAGE_CLASS
#define TCL_STORAGE_CLASS DLLEXPORT

TdomStubs *tdomStubsPtr;

/*----------------------------------------------------------------------------
|   Tdom_InitStubs
|
\---------------------------------------------------------------------------*/

CONST char *
Tdom_InitStubs (
    Tcl_Interp *interp, 
    char *version, 
    int exact
    )
{
    CONST char *actualVersion;
    ClientData clientData = NULL;

#if (TCL_MAJOR_VERSION == 8) && (TCL_MINOR_VERSION == 0)
    Tcl_SetResult(interp, "Too old Tcl version. Binary extensions "
                  "to tDOM are not possible, with a that outdated "
                  "Tcl version.", TCL_STATIC);
    return NULL;
#else
    actualVersion = Tcl_PkgRequireEx(interp, "tdom", version, exact,
                                     (ClientData*) &clientData);
    tdomStubsPtr = (TdomStubs*)clientData;

    if (!actualVersion) {
        return NULL;
    }
    if (!tdomStubsPtr) {
        Tcl_SetResult(interp, "This implementation of Tdom does not "
                      "support stubs", TCL_STATIC);
        return NULL;
    }
    
    return actualVersion;
#endif
}
