/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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
#include "tclExtend.h"
#include "tk.h"

/*
 * The following variable is a special hack that insures the tcl
 * version of matherr() is used when linking against shared libraries
 * Even if matherr is not used on this system, there is a dummy version
 * in libtcl.
 */
EXTERN int matherr ();
int (*tclDummyMathPtr)() = matherr;


/*-----------------------------------------------------------------------------
 * main --
 *
 * This is the main program for the application.
 *-----------------------------------------------------------------------------
 */
#ifdef __cplusplus
int
main (int    argc,
      char **argv)
#else
int
main (argc, argv)
    int    argc;
    char **argv;
#endif
{
#ifdef USE_TCLX
    TkX_Main(argc, argv, Tcl_AppInit);
#else
    Tk_Main(argc, argv, Tcl_AppInit);
#endif
    return 0;                   /* Needed only to prevent compiler warning. */
}

/*-----------------------------------------------------------------------------
 * Tcl_AppInit --
 *
 * This procedure performs application-specific initialization. Most
 * applications, especially those that incorporate additional packages, will
 * have their own version of this procedure.
 *
 * Results:
 *    Returns a standard Tcl completion code, and leaves an error message in
 * interp->result if an error occurs.
 *-----------------------------------------------------------------------------
 */
#ifdef __cplusplus
int
Tcl_AppInit (Tcl_Interp *interp)
#else
int
Tcl_AppInit (interp)
    Tcl_Interp *interp;
#endif
{
    if (Tcl_Init (interp) == TCL_ERROR) {
        return TCL_ERROR;
    }
#ifdef USE_TCLX
    if (Tclx_Init(interp) == TCL_ERROR) {
        return TCL_ERROR;
    }
    Tcl_StaticPackage(interp, "Tclx", Tclx_Init, Tclx_SafeInit);
#endif
    if (Tk_Init(interp) == TCL_ERROR) {
        return TCL_ERROR;
    }
    Tcl_StaticPackage(interp, "Tk", Tk_Init, (Tcl_PackageInitProc *) NULL);
#ifdef USE_TCLX
    if (Tkx_Init(interp) == TCL_ERROR) {
        return TCL_ERROR;
    }
    Tcl_StaticPackage(interp, "Tkx", Tkx_Init, (Tcl_PackageInitProc *) NULL);
#endif

    if (Ldaptcl_Init(interp) == TCL_ERROR) {
	return TCL_ERROR;
    }
    Tcl_StaticPackage(interp, "Ldaptcl", Ldaptcl_Init,
            (Tcl_PackageInitProc *) NULL);

    /*
     * Call Tcl_CreateCommand for application-specific commands, if
     * they weren't already created by the init procedures called above.
     */

    /*
     * Specify a user-specific startup file to invoke if the application
     * is run interactively.  Typically the startup file is "~/.apprc"
     * where "app" is the name of the application.  If this line is deleted
     * then no user-specific startup file will be run under any conditions.
     */
    Tcl_SetVar(interp, "tcl_rcFileName", "~/.wishxrc", TCL_GLOBAL_ONLY);
    return TCL_OK;
}
