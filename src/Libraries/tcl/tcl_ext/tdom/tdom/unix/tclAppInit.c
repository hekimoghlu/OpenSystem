/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 23, 2024.
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
#include "tcl.h"
 
extern int Tdom_Init _ANSI_ARGS_((Tcl_Interp *interp));
extern int Tdom_SafeInit _ANSI_ARGS_((Tcl_Interp *interp));

/*----------------------------------------------------------------------------
|   main
|
\---------------------------------------------------------------------------*/
int
main(
    int    argc,
    char **argv
    )
{
    Tcl_Main (argc, argv, Tcl_AppInit);
    return 0;
}

/*----------------------------------------------------------------------------
|   Tcl_AppInit
|
\---------------------------------------------------------------------------*/
int
Tcl_AppInit(interp)
    Tcl_Interp *interp;
{
    if (Tcl_Init(interp) == TCL_ERROR) {
        return TCL_ERROR;
    }
    if (Tdom_Init(interp) == TCL_ERROR) {
        return TCL_ERROR;
    }
    Tcl_StaticPackage(interp, "tdom", Tdom_Init, Tdom_SafeInit);
    Tcl_SetVar(interp, "tcl_rcFileName", "~/.tcldomshrc", TCL_GLOBAL_ONLY);
    return TCL_OK;
}
