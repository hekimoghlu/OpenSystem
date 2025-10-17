/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 31, 2021.
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
#include "tkUnixInt.h"

#ifdef HAVE_COREFOUNDATION
static int		GetLibraryPath(Tcl_Interp *interp);
#else
#define GetLibraryPath(dummy)	(void)0
#endif /* HAVE_COREFOUNDATION */

/*
 *----------------------------------------------------------------------
 *
 * TkpInit --
 *
 *	Performs Unix-specific interpreter initialization related to the
 *      tk_library variable.
 *
 * Results:
 *	Returns a standard Tcl result. Leaves an error message or result in
 *	the interp's result.
 *
 * Side effects:
 *	Sets "tk_library" Tcl variable, runs "tk.tcl" script.
 *
 *----------------------------------------------------------------------
 */

int
TkpInit(
    Tcl_Interp *interp)
{
    TkCreateXEventSource();
    GetLibraryPath(interp);
    return TCL_OK;
}

/*
 *----------------------------------------------------------------------
 *
 * TkpGetAppName --
 *
 *	Retrieves the name of the current application from a platform specific
 *	location. For Unix, the application name is the tail of the path
 *	contained in the tcl variable argv0.
 *
 * Results:
 *	Returns the application name in the given Tcl_DString.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

void
TkpGetAppName(
    Tcl_Interp *interp,
    Tcl_DString *namePtr)	/* A previously initialized Tcl_DString. */
{
    CONST char *p, *name;

    name = Tcl_GetVar(interp, "argv0", TCL_GLOBAL_ONLY);
    if ((name == NULL) || (*name == 0)) {
	name = "tk";
    } else {
	p = strrchr(name, '/');
	if (p != NULL) {
	    name = p+1;
	}
    }
    Tcl_DStringAppend(namePtr, name, -1);
}

/*
 *----------------------------------------------------------------------
 *
 * TkpDisplayWarning --
 *
 *	This routines is called from Tk_Main to display warning messages that
 *	occur during startup.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Generates messages on stdout.
 *
 *----------------------------------------------------------------------
 */

void
TkpDisplayWarning(
    CONST char *msg,		/* Message to be displayed. */
    CONST char *title)		/* Title of warning. */
{
    Tcl_Channel errChannel = Tcl_GetStdChannel(TCL_STDERR);
    if (errChannel) {
	Tcl_WriteChars(errChannel, title, -1);
	Tcl_WriteChars(errChannel, ": ", 2);
	Tcl_WriteChars(errChannel, msg, -1);
	Tcl_WriteChars(errChannel, "\n", 1);
    }
}

#ifdef HAVE_COREFOUNDATION

/*
 *----------------------------------------------------------------------
 *
 * GetLibraryPath --
 *
 *	If we have a bundle structure for the Tk installation, then check
 *	there first to see if we can find the libraries there.
 *
 * Results:
 *	TCL_OK if we have found the tk library; TCL_ERROR otherwise.
 *
 * Side effects:
 *	Same as for Tcl_MacOSXOpenVersionedBundleResources.
 *
 *----------------------------------------------------------------------
 */

static int
GetLibraryPath(
    Tcl_Interp *interp)
{
#ifdef TK_FRAMEWORK
    int foundInFramework = TCL_ERROR;
    char tkLibPath[PATH_MAX + 1];

    foundInFramework = Tcl_MacOSXOpenVersionedBundleResources(interp,
	    "com.tcltk.tklibrary", TK_FRAMEWORK_VERSION, 0, PATH_MAX,
	    tkLibPath);
    if (tkLibPath[0] != '\0') {
        Tcl_SetVar(interp, "tk_library", tkLibPath, TCL_GLOBAL_ONLY);
    }
    return foundInFramework;
#else
    return TCL_ERROR;
#endif
}
#endif /* HAVE_COREFOUNDATION */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
