/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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
#define USE_TCL_STUBS
#endif

#include "memchan.h"
#include "buf.h"

#undef  TCL_STORAGE_CLASS
#define TCL_STORAGE_CLASS DLLEXPORT

extern BufStubs *bufStubsPtr;
MemchanStubs *memchanStubsPtr;

/*
 *----------------------------------------------------------------------
 *
 * Memchan_InitStubs --
 *
 *	Loads the Memchan extension and initializes the stubs table.
 *
 * Results:
 *	The actual version of Memchan in use. NULL if an error occurred.
 *
 * Side effects:
 *	Sets the stub table pointers.
 *
 *----------------------------------------------------------------------
 */

CONST char *
Memchan_InitStubs(interp, version, exact)
    Tcl_Interp *interp;
    CONST char *version;
    int exact;
{
    CONST char *result;

    /* HACK: de-CONST 'version' if compiled against 8.3.
     * The API has no CONST despite not modifying the argument
     * And a debug build with high warning-level on windows
     * will abort the compilation.
     */

#if ((TCL_MAJOR_VERSION < 8) || ((TCL_MAJOR_VERSION == 8) && (TCL_MINOR_VERSION < 4)))
#define UNCONST (char*)
#else
#define UNCONST 
#endif

    /* NOTE: Memchan actuall provide the Buf stubs. The Memchan stubs
     *       table is hooked into this.
     */

    result = Tcl_PkgRequireEx(interp, "Memchan", UNCONST version, exact,
		(ClientData *) &bufStubsPtr);
    if (!result || !bufStubsPtr) {
        return (char *) NULL;
    }

    memchanStubsPtr = bufStubsPtr->hooks->memchanStubs;
    return result;
}
#undef UNCONST
