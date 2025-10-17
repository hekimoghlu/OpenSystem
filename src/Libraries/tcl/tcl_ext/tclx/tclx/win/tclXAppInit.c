/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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
 * As a shell (i.e., the main program) we cannot be using the stubs table.
 */
#ifdef USE_TCL_STUBS
#undef USE_TCL_STUBS
#endif

#include "tclExtend.h"

/*-----------------------------------------------------------------------------
 * TclX_AppInit --
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
TclX_AppInit (Tcl_Interp *interp)
{
    if (Tcl_Init (interp) == TCL_ERROR) {
        return TCL_ERROR;
    }
    if (Tclx_Init (interp) == TCL_ERROR) {
        return TCL_ERROR;
    }
    Tcl_StaticPackage (interp, "Tclx", Tclx_Init, Tclx_SafeInit);

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
    Tcl_SetVar(interp, "tcl_rcFileName", "~/.tclrc", TCL_GLOBAL_ONLY);
    return TCL_OK;
}


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
    TclX_MainEx (argc, argv, TclX_AppInit, Tcl_CreateInterp());

    return 0;                   /* Needed only to prevent compiler warning. */
}

