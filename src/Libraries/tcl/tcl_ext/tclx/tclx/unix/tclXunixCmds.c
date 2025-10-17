/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 22, 2023.
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

static int 
TclX_ChrootObjCmd _ANSI_ARGS_((ClientData clientData,
                              Tcl_Interp *interp, 
			      int         objc,
			      Tcl_Obj     *CONST objv[]));

static int 
TclX_TimesObjCmd _ANSI_ARGS_((ClientData   clientData,
                             Tcl_Interp  *interp,
			     int          objc,
			     Tcl_Obj      *CONST objv[]));


/*-----------------------------------------------------------------------------
 * TclX_ChrootObjCmd --
 *     Implements the TCL chroot command:
 *         chroot path
 *
 * Results:
 *      Standard TCL results, may return the UNIX system error message.
 *
 *-----------------------------------------------------------------------------
 */
static int
TclX_ChrootObjCmd (clientData, interp, objc, objv)
       ClientData  clientData;
       Tcl_Interp *interp;
       int         objc;
       Tcl_Obj   *CONST objv[];
{
    char   *chrootString;
    int     chrootStrLen;

    if (objc != 2)
	return TclX_WrongArgs (interp, objv [0], "path");

    chrootString = Tcl_GetStringFromObj (objv [1], &chrootStrLen);

    if (chroot (chrootString) < 0) {
        TclX_AppendObjResult (interp, "changing root to \"", chrootString,
                              "\" failed: ", Tcl_PosixError (interp),
                              (char *) NULL);
        return TCL_ERROR;
    }
    return TCL_OK;
}

/*-----------------------------------------------------------------------------
 * TclX_TimesObjCmd --
 *     Implements the TCL times command:
 *     times
 *
 * Results:
 *  Standard TCL results.
 *
 *-----------------------------------------------------------------------------
 */
static int
TclX_TimesObjCmd (clientData, interp, objc, objv)
    ClientData  clientData;
    Tcl_Interp *interp;
    int         objc;
    Tcl_Obj   *CONST objv[];
{
    struct tms tm;
    char       timesBuf [48];

    if (objc != 1)
	return TclX_WrongArgs (interp, objv [0], "");

    times (&tm);

    sprintf (timesBuf, "%ld %ld %ld %ld", 
             (long) TclXOSTicksToMS (tm.tms_utime),
             (long) TclXOSTicksToMS (tm.tms_stime),
             (long) TclXOSTicksToMS (tm.tms_cutime),
             (long) TclXOSTicksToMS (tm.tms_cstime));

    Tcl_SetStringObj (Tcl_GetObjResult (interp), timesBuf, -1);
    return TCL_OK;
}


/*-----------------------------------------------------------------------------
 * TclX_PlatformCmdsInit --
 *     Initialize the platform-specific commands.
 *-----------------------------------------------------------------------------
 */
void
TclX_PlatformCmdsInit (interp)
    Tcl_Interp *interp;
{
    Tcl_CreateObjCommand (interp,
			  "chroot",
			  TclX_ChrootObjCmd,
                          (ClientData) NULL,
			  (Tcl_CmdDeleteProc *) NULL);

    Tcl_CreateObjCommand (interp, 
			  "times",
			  TclX_TimesObjCmd,
                          (ClientData) NULL,
			  (Tcl_CmdDeleteProc*) NULL);
    
}

