/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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

extern int
Tcltest_Init (Tcl_Interp *interp);

extern int
Tclxtest_Init (Tcl_Interp *interp);


/*-----------------------------------------------------------------------------
 * main --
 *
 * This is the main program for the application.
 *-----------------------------------------------------------------------------
 */
int
main (int    argc,
      char **argv)
{
    TclX_Main (argc, argv, Tcl_AppInit);
    return 0;                   /* Needed only to prevent compiler warning. */
}


/*-----------------------------------------------------------------------------
 * Tcl_AppInit --
 *
 * This procedure performs application-specific initialization.  Most
 * applications, especially those that incorporate additional packages, will
 * have their own version of this procedure.
 *
 * Results:
 *   Returns a standard Tcl completion code, and leaves an error message in
 *  interp result if an error occurs.
 *-----------------------------------------------------------------------------
 */
int
Tcl_AppInit (Tcl_Interp *interp)
{
    if (Tcl_Init (interp) == TCL_ERROR) {
        return TCL_ERROR;
    }

    if (Tclx_Init (interp) == TCL_ERROR) {
        return TCL_ERROR;
    }
    Tcl_StaticPackage (interp, "Tclx", Tclx_Init, Tclx_SafeInit);

    if (Tcltest_Init (interp) == TCL_ERROR) {
	return TCL_ERROR;
    }
    Tcl_StaticPackage(interp, "Tcltest", Tcltest_Init,
                      (Tcl_PackageInitProc *) NULL);

    if (Tclxtest_Init (interp) == TCL_ERROR) {
	return TCL_ERROR;
    }
    Tcl_StaticPackage(interp, "Tclxtest", Tclxtest_Init,
                      (Tcl_PackageInitProc *) NULL);

    return TCL_OK;
}


