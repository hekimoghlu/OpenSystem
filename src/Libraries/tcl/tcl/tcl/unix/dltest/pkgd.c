/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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

/*
 * Prototypes for procedures defined later in this file:
 */

static int    Pkgd_SubObjCmd(ClientData clientData,
		Tcl_Interp *interp, int objc, Tcl_Obj *CONST objv[]);
static int    Pkgd_UnsafeObjCmd(ClientData clientData,
		Tcl_Interp *interp, int objc, Tcl_Obj *CONST objv[]);

/*
 *----------------------------------------------------------------------
 *
 * Pkgd_SubObjCmd --
 *
 *	This procedure is invoked to process the "pkgd_sub" Tcl command. It
 *	expects two arguments and returns their difference.
 *
 * Results:
 *	A standard Tcl result.
 *
 * Side effects:
 *	See the user documentation.
 *
 *----------------------------------------------------------------------
 */

static int
Pkgd_SubObjCmd(
    ClientData dummy,		/* Not used. */
    Tcl_Interp *interp,		/* Current interpreter. */
    int objc,			/* Number of arguments. */
    Tcl_Obj *CONST objv[])	/* Argument objects. */
{
    int first, second;

    if (objc != 3) {
	Tcl_WrongNumArgs(interp, 1, objv, "num num");
	return TCL_ERROR;
    }
    if ((Tcl_GetIntFromObj(interp, objv[1], &first) != TCL_OK)
	    || (Tcl_GetIntFromObj(interp, objv[2], &second) != TCL_OK)) {
	return TCL_ERROR;
    }
    Tcl_SetObjResult(interp, Tcl_NewIntObj(first - second));
    return TCL_OK;
}

/*
 *----------------------------------------------------------------------
 *
 * Pkgd_UnsafeCmd --
 *
 *	This procedure is invoked to process the "pkgd_unsafe" Tcl command. It
 *	just returns a constant string.
 *
 * Results:
 *	A standard Tcl result.
 *
 * Side effects:
 *	See the user documentation.
 *
 *----------------------------------------------------------------------
 */

static int
Pkgd_UnsafeObjCmd(
    ClientData dummy,		/* Not used. */
    Tcl_Interp *interp,		/* Current interpreter. */
    int objc,			/* Number of arguments. */
    Tcl_Obj *CONST objv[])	/* Argument objects. */
{
    Tcl_SetObjResult(interp, Tcl_NewStringObj("unsafe command invoked", -1));
    return TCL_OK;
}

/*
 *----------------------------------------------------------------------
 *
 * Pkgd_Init --
 *
 *	This is a package initialization procedure, which is called by Tcl
 *	when this package is to be added to an interpreter.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

int
Pkgd_Init(
    Tcl_Interp *interp)		/* Interpreter in which the package is to be
				 * made available. */
{
    int code;

    if (Tcl_InitStubs(interp, TCL_VERSION, 0) == NULL) {
	return TCL_ERROR;
    }
    code = Tcl_PkgProvide(interp, "Pkgd", "7.3");
    if (code != TCL_OK) {
	return code;
    }
    Tcl_CreateObjCommand(interp, "pkgd_sub", Pkgd_SubObjCmd,
	    (ClientData) 0, (Tcl_CmdDeleteProc *) NULL);
    Tcl_CreateObjCommand(interp, "pkgd_unsafe", Pkgd_UnsafeObjCmd,
	    (ClientData) 0, (Tcl_CmdDeleteProc *) NULL);
    return TCL_OK;
}

/*
 *----------------------------------------------------------------------
 *
 * Pkgd_SafeInit --
 *
 *	This is a package initialization procedure, which is called by Tcl
 *	when this package is to be added to a safe interpreter.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

int
Pkgd_SafeInit(
    Tcl_Interp *interp)		/* Interpreter in which the package is to be
				 * made available. */
{
    int code;

    if (Tcl_InitStubs(interp, TCL_VERSION, 0) == NULL) {
	return TCL_ERROR;
    }
    code = Tcl_PkgProvide(interp, "Pkgd", "7.3");
    if (code != TCL_OK) {
	return code;
    }
    Tcl_CreateObjCommand(interp, "pkgd_sub", Pkgd_SubObjCmd, (ClientData) 0,
	    (Tcl_CmdDeleteProc *) NULL);
    return TCL_OK;
}
