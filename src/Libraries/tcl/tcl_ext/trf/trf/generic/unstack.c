/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 25, 2023.
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
#include	"transformInt.h"

static int
TrfUnstackObjCmd _ANSI_ARGS_ ((ClientData notUsed, Tcl_Interp* interp,
			       int objc, struct Tcl_Obj* CONST * objv));

/*
 *----------------------------------------------------------------------
 *
 * TrfUnstackCmd --
 *
 *	This procedure is invoked to process the "unstack" Tcl command.
 *	See the user documentation for details on what it does.
 *
 * Results:
 *	A standard Tcl result.
 *
 * Side effects:
 *	Unstacks the channel, thereby restoring its parent.
 *
 *----------------------------------------------------------------------
 */

static int
TrfUnstackObjCmd (notUsed, interp, objc, objv)
     ClientData  notUsed;		/* Not used. */
     Tcl_Interp* interp;		/* Current interpreter. */
     int                     objc;	/* Number of arguments. */
     struct Tcl_Obj* CONST * objv;	/* Argument strings. */
{
  /*
   * unstack <channel>
   */

  Tcl_Channel chan;
  int         mode;

#ifdef USE_TCL_STUBS
  if (Tcl_UnstackChannel == NULL) {
    const char* cmd = Tcl_GetStringFromObj (objv [0], NULL);

    Tcl_AppendResult (interp, cmd, " is not available as the required ",
		      "patch to the core was not applied", (char*) NULL);
    return TCL_ERROR;
  }
#endif

  if ((objc < 2) || (objc > 2)) {
    Tcl_AppendResult (interp,
		      "wrong # args: should be \"unstack channel\"",
		      (char*) NULL);
    return TCL_ERROR;
  }

  chan = Tcl_GetChannel (interp, Tcl_GetStringFromObj (objv [1], NULL), &mode);

  if (chan == (Tcl_Channel) NULL) {
    return TCL_ERROR;
  }

  Tcl_UnstackChannel (interp, chan);
  return TCL_OK;
}

/*
 *------------------------------------------------------*
 *
 *	TrfInit_Unstack --
 *
 *	------------------------------------------------*
 *	Register the 'unstack' command.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		As of 'Tcl_CreateObjCommand'.
 *
 *	Result:
 *		A standard Tcl error code.
 *
 *------------------------------------------------------*
 */

int
TrfInit_Unstack (interp)
Tcl_Interp* interp;
{
  Tcl_CreateObjCommand (interp, "unstack", TrfUnstackObjCmd,
			(ClientData) NULL,
			(Tcl_CmdDeleteProc *) NULL);

  return TCL_OK;
}

