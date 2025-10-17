/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 17, 2022.
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
#include "tclExtdInt.h"

/*
 * Prototypes of internal functions.
 */
static int
IdProcess  _ANSI_ARGS_((Tcl_Interp *interp,
			int objc,
			Tcl_Obj *CONST objv[]));

static int
IdHost _ANSI_ARGS_((Tcl_Interp *interp,
		    int objc,
		    Tcl_Obj *CONST objv[]));

static int 
TclX_IdObjCmd _ANSI_ARGS_((ClientData clientData,
                           Tcl_Interp *interp,
                           int objc,
                           Tcl_Obj *CONST objv[]));

/*-----------------------------------------------------------------------------
 * Tcl_IdCmd --
 *     Implements the TclX id command on Win32.
 *
 *        id host
 *        id process
 *
 * Results:
 *  Standard TCL results, may return the Posix system error message.
 *
 *-----------------------------------------------------------------------------
 */

/*
 * id process
 */
static int
IdProcess (interp, objc, objv)
    Tcl_Interp *interp;
    int         objc;
    Tcl_Obj    *CONST objv[];
{
    Tcl_Obj *resultPtr = Tcl_GetObjResult (interp);

    if (objc != 2) {
        TclX_AppendObjResult (interp, tclXWrongArgs, objv [0], 
                              " process", (char *) NULL);
        return TCL_ERROR;
    }
    Tcl_SetLongObj (resultPtr, getpid());
    return TCL_OK;
}

/*
 * id host
 */
static int
IdHost (interp, objc, objv)
    Tcl_Interp *interp;
    int         objc;
    Tcl_Obj    *CONST objv[];
{
    char hostName [TCL_RESULT_SIZE];

    if (objc != 2) {
        TclX_AppendObjResult (interp, tclXWrongArgs, objv [0], 
                              " host", (char *) NULL);
        return TCL_ERROR;
    }
    if (gethostname (hostName, sizeof (hostName)) < 0) {
        TclX_AppendObjResult (interp, "failed to get host name: ",
                              Tcl_PosixError (interp), (char *) NULL);
        return TCL_ERROR;
    }
    TclX_AppendObjResult (interp, hostName, (char *) NULL);
    return TCL_OK;
}

static int
TclX_IdObjCmd (clientData, interp, objc, objv)
    ClientData  clientData;
    Tcl_Interp *interp;
    int         objc;
    Tcl_Obj    *CONST objv[];
{
    char *optionPtr;

    if (objc < 2) {
        TclX_AppendObjResult (interp, tclXWrongArgs, objv [0], " arg ?arg...?",
                              (char *) NULL);
        return TCL_ERROR;
    }

    optionPtr = Tcl_GetStringFromObj (objv[1], NULL);

    /*
     * If the first argument is "process", return the process ID, parent's
     * process ID, process group or set the process group depending on args.
     */
    if (STREQU (optionPtr, "process")) {
        return IdProcess (interp, objc, objv);
    }

    /*
     * Handle returning the host name if its available.
     */
    if (STREQU (optionPtr, "host")) {
        return IdHost (interp, objc, objv);
    }

    TclX_AppendObjResult (interp, "second arg must be one of \"process\", ",
                          "or \"host\"", (char *) NULL);
    return TCL_ERROR;
}


/*-----------------------------------------------------------------------------
 * TclX_IdInit --
 *     Initialize the id command.
 *-----------------------------------------------------------------------------
 */
void
TclX_IdInit (interp)
    Tcl_Interp *interp;
{
    Tcl_CreateObjCommand (interp,
			  "id",
			  TclX_IdObjCmd,
                          (ClientData) NULL,
			  (Tcl_CmdDeleteProc*) NULL);
}

