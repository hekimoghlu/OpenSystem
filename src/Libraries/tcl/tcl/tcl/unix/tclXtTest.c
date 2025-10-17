/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 20, 2025.
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
#include <X11/Intrinsic.h>
#include "tcl.h"

static int	TesteventloopCmd(ClientData clientData,
		    Tcl_Interp *interp, int argc, CONST char **argv);
extern void	InitNotifier(void);

/*
 *----------------------------------------------------------------------
 *
 * Tclxttest_Init --
 *
 *	This procedure performs application-specific initialization. Most
 *	applications, especially those that incorporate additional packages,
 *	will have their own version of this procedure.
 *
 * Results:
 *	Returns a standard Tcl completion code, and leaves an error message in
 *	the interp's result if an error occurs.
 *
 * Side effects:
 *	Depends on the startup script.
 *
 *----------------------------------------------------------------------
 */

int
Tclxttest_Init(
    Tcl_Interp *interp)		/* Interpreter for application. */
{
    if (Tcl_InitStubs(interp, TCL_VERSION, 0) == NULL) {
	return TCL_ERROR;
    }
    XtToolkitInitialize();
    InitNotifier();
    Tcl_CreateCommand(interp, "testeventloop", TesteventloopCmd,
            (ClientData) 0, NULL);
    return TCL_OK;
}

/*
 *----------------------------------------------------------------------
 *
 * TesteventloopCmd --
 *
 *	This procedure implements the "testeventloop" command. It is used to
 *	test the Tcl notifier from an "external" event loop (i.e. not
 *	Tcl_DoOneEvent()).
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
TesteventloopCmd(
    ClientData clientData,	/* Not used. */
    Tcl_Interp *interp,		/* Current interpreter. */
    int argc,			/* Number of arguments. */
    CONST char **argv)		/* Argument strings. */
{
    static int *framePtr = NULL;/* Pointer to integer on stack frame of
				 * innermost invocation of the "wait"
				 * subcommand. */

   if (argc < 2) {
	Tcl_AppendResult(interp, "wrong # arguments: should be \"", argv[0],
                " option ... \"", NULL);
        return TCL_ERROR;
    }
    if (strcmp(argv[1], "done") == 0) {
	*framePtr = 1;
    } else if (strcmp(argv[1], "wait") == 0) {
	int *oldFramePtr;
	int done;
	int oldMode = Tcl_SetServiceMode(TCL_SERVICE_ALL);

	/*
	 * Save the old stack frame pointer and set up the current frame.
	 */

	oldFramePtr = framePtr;
	framePtr = &done;

	/*
	 * Enter an Xt event loop until the flag changes. Note that we do not
	 * explicitly call Tcl_ServiceEvent().
	 */

	done = 0;
	while (!done) {
	    XtAppProcessEvent(TclSetAppContext(NULL), XtIMAll);
	}
	(void) Tcl_SetServiceMode(oldMode);
	framePtr = oldFramePtr;
    } else {
	Tcl_AppendResult(interp, "bad option \"", argv[1],
		"\": must be done or wait", NULL);
	return TCL_ERROR;
    }
    return TCL_OK;
}
