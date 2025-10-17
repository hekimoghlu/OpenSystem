/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 8, 2023.
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
#include <sys/types.h>
#include <loader.h>

/*
 *----------------------------------------------------------------------
 *
 * TclpDlopen --
 *
 *	Dynamically loads a binary code file into memory and returns a handle
 *	to the new code.
 *
 * Results:
 *	A standard Tcl completion code.  If an error occurs, an error message
 *	is left in the interp's result.
 *
 * Side effects:
 *	New code suddenly appears in memory.
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
    ldr_module_t lm;
    char *pkg;
    char *fileName = Tcl_GetString(pathPtr);
    CONST char *native;

    /*
     * First try the full path the user gave us.  This is particularly
     * important if the cwd is inside a vfs, and we are trying to load using a
     * relative path.
     */

    native = Tcl_FSGetNativePath(pathPtr);
    lm = (Tcl_PackageInitProc *) load(native, LDR_NOFLAGS);

    if (lm == LDR_NULL_MODULE) {
	/*
	 * Let the OS loader examine the binary search path for whatever
	 * string the user gave us which hopefully refers to a file on the
	 * binary path
	 */

	Tcl_DString ds;

	native = Tcl_UtfToExternalDString(NULL, fileName, -1, &ds);
	lm = (Tcl_PackageInitProc *) load(native, LDR_NOFLAGS);
	Tcl_DStringFree(&ds);
    }

    if (lm == LDR_NULL_MODULE) {
	Tcl_AppendResult(interp, "couldn't load file \"", fileName, "\": ",
		Tcl_PosixError(interp), NULL);
	return TCL_ERROR;
    }

    *clientDataPtr = NULL;

    /*
     * My convention is to use a [OSF loader] package name the same as shlib,
     * since the idiots never implemented ldr_lookup() and it is otherwise
     * impossible to get a package name given a module.
     *
     * I build loadable modules with a makefile rule like
     *		ld ... -export $@: -o $@ $(OBJS)
     */

    if ((pkg = strrchr(fileName, '/')) == NULL) {
        pkg = fileName;
    } else {
	pkg++;
    }
    *loadHandle = pkg;
    *unloadProcPtr = &TclpUnloadFile;
    return TCL_OK;
}

/*
 *----------------------------------------------------------------------
 *
 * TclpFindSymbol --
 *
 *	Looks up a symbol, by name, through a handle associated with a
 *	previously loaded piece of code (shared library).
 *
 * Results:
 *	Returns a pointer to the function associated with 'symbol' if it is
 *	found.  Otherwise returns NULL and may leave an error message in the
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
    return ldr_lookup_package((char *)loadHandle, symbol);
}

/*
 *----------------------------------------------------------------------
 *
 * TclpUnloadFile --
 *
 *	Unloads a dynamically loaded binary code file from memory. Code
 *	pointers in the formerly loaded file are no longer valid after calling
 *	this function.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Does nothing.  Can anything be done?
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
 *----------------------------------------------------------------------
 *
 * TclGuessPackageName --
 *
 *	If the "load" command is invoked without providing a package name,
 *	this function is invoked to try to figure it out.
 *
 * Results:
 *	Always returns 0 to indicate that we couldn't figure out a package
 *	name; generic code will then try to guess the package from the file
 *	name.  A return value of 1 would have meant that we figured out the
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
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
