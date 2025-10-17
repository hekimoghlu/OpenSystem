/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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
#include "tclInt.h"

/*
 *----------------------------------------------------------------------
 *
 * TclpDlopen --
 *
 *	This procedure is called to carry out dynamic loading of binary code;
 *	it is intended for use only on systems that don't support dynamic
 *	loading (it returns an error).
 *
 * Results:
 *	The result is TCL_ERROR, and an error message is left in the interp's
 *	result.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

int
TclpDlopen(
    Tcl_Interp *interp,		/* Used for error reporting. */
    Tcl_Obj *pathPtr,		/* Name of the file containing the desired
				 * code (UTF-8). */
    Tcl_LoadHandle *loadHandle,	/* Filled with token for dynamically loaded
				 * file which will be passed back to
				 * (*unloadProcPtr)() to unload the file. */
    Tcl_FSUnloadFileProc **unloadProcPtr)
				/* Filled with address of Tcl_FSUnloadFileProc
				 * function which should be used for this
				 * file. */
{
    Tcl_SetResult(interp,
	    "dynamic loading is not currently available on this system",
	    TCL_STATIC);
    return TCL_ERROR;
}

/*
 *----------------------------------------------------------------------
 *
 * TclpFindSymbol --
 *
 *	Looks up a symbol, by name, through a handle associated with a
 *	previously loaded piece of code (shared library). This version of this
 *	routine should never be called because the associated TclpDlopen()
 *	function always returns an error.
 *
 * Results:
 *	Returns a pointer to the function associated with 'symbol' if it is
 *	found. Otherwise returns NULL and may leave an error message in the
 *	interp's result.
 *
 *----------------------------------------------------------------------
 */

Tcl_PackageInitProc *
TclpFindSymbol(
    Tcl_Interp *interp,
    Tcl_LoadHandle loadHandle,
    CONST char *symbol)
{
    return NULL;
}

/*
 *----------------------------------------------------------------------
 *
 * TclGuessPackageName --
 *
 *	If the "load" command is invoked without providing a package name,
 *	this procedure is invoked to try to figure it out.
 *
 * Results:
 *	Always returns 0 to indicate that we couldn't figure out a package
 *	name; generic code will then try to guess the package from the file
 *	name. A return value of 1 would have meant that we figured out the
 *	package name and put it in bufPtr.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

int
TclGuessPackageName(
    CONST char *fileName,	/* Name of file containing package (already
				 * translated to local form if needed). */
    Tcl_DString *bufPtr)	/* Initialized empty dstring. Append package
				 * name to this if possible. */
{
    return 0;
}

/*
 *----------------------------------------------------------------------
 *
 * TclpUnloadFile --
 *
 *    This procedure is called to carry out dynamic unloading of binary code;
 *    it is intended for use only on systems that don't support dynamic
 *    loading (it does nothing).
 *
 * Results:
 *    None.
 *
 * Side effects:
 *    None.
 *
 *----------------------------------------------------------------------
 */

void
TclpUnloadFile(
    Tcl_LoadHandle loadHandle)	/* loadHandle returned by a previous call to
				 * TclpDlopen(). The loadHandle is a token
				 * that represents the loaded file. */
{
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
