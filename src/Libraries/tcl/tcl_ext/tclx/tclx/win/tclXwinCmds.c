/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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
 * Tcl_ChrootObjCmd --
 *   Stub to return an error if the chroot command is used on Windows.
 *-----------------------------------------------------------------------------
 */
static int
TclX_ChrootObjCmd (ClientData  clientData,
                  Tcl_Interp *interp,
                  int         objc,
                  Tcl_Obj   *CONST objv[])
{
    return TclXNotAvailableObjError (interp, objv [0]);
}

/*-----------------------------------------------------------------------------
 * Tcl_TimesObjCmd --
 *   Stub to return an error if the times command is used on Windows.
 *-----------------------------------------------------------------------------
 */
static int
TclX_TimesObjCmd (ClientData  clientData,
                 Tcl_Interp *interp,
                 int         objc,
                 Tcl_Obj   *CONST objv[])
{
    return TclXNotAvailableObjError (interp, objv [0]);
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


/*-----------------------------------------------------------------------------
 * TclX_ServerInit --
 *     
 *   Stub, does nothing.  The Unix version of the function initilizes some
 * compatiblity functions that are not implemented on Win32.
 *-----------------------------------------------------------------------------
 */
void
TclX_ServerInit (Tcl_Interp *interp)
{
}
