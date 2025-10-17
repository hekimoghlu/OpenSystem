/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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
#include "tkMacOSXPrivate.h"

/*
 * Forward declarations of procedures defined later in this file:
 */

static int		DebuggerCmd (ClientData dummy, Tcl_Interp *interp,
			    int argc, const char **argv);
MODULE_SCOPE int	TkplatformtestInit(Tcl_Interp *interp);

/*
 *----------------------------------------------------------------------
 *
 * TkplatformtestInit --
 *
 *	Defines commands that test platform specific functionality for
 *	Unix platforms.
 *
 * Results:
 *	A standard Tcl result.
 *
 * Side effects:
 *	Defines new commands.
 *
 *----------------------------------------------------------------------
 */

int
TkplatformtestInit(
    Tcl_Interp *interp)		/* Interpreter to add commands to. */
{
    /*
     * Add commands for platform specific tests on MacOS here.
     */

    Tcl_CreateCommand(interp, "debugger", DebuggerCmd,
	    (ClientData) 0, (Tcl_CmdDeleteProc *) NULL);

    return TCL_OK;
}

/*
 *----------------------------------------------------------------------
 *
 * DebuggerCmd --
 *
 *	This procedure simply calls the low level debugger.
 *
 * Results:
 *	A standard Tcl result.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

static int
DebuggerCmd(
    ClientData clientData,		/* Not used. */
    Tcl_Interp *interp,			/* Not used. */
    int argc,				/* Not used. */
    const char **argv)			/* Not used. */
{
    Debugger();
    return TCL_OK;
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 79
 * coding: utf-8
 * End:
 */
