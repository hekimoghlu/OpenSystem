/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 18, 2022.
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
#include "tkUnixInt.h"

/*
 * The wrapper code for Unix is actually set up in library/tk.tcl these days;
 * the procedure names used here are probably wrong too...
 */

#ifdef TK_OBSOLETE_UNIX_DIALOG_WRAPPERS

/*
 *----------------------------------------------------------------------
 *
 * EvalObjv --
 *
 *	Invokes the Tcl procedure with the arguments.
 *
 * Results:
 *	Returns the result of the evaluation of the command.
 *
 * Side effects:
 *	The command may be autoloaded.
 *
 *----------------------------------------------------------------------
 */

static int
EvalObjv(
    Tcl_Interp *interp,		/* Current interpreter. */
    char *cmdName,		/* Name of the TCL command to call */
    int objc,			/* Number of arguments. */
    Tcl_Obj *CONST *objv)	/* Arguments. */
{
    Tcl_Obj *cmdObj, **objs;
    int result;

    cmdObj = Tcl_NewStringObj(cmdName, -1);
    Tcl_IncrRefCount(cmdObj);
    objs = (Tcl_Obj **) ckalloc(sizeof(Tcl_Obj*) * (unsigned)(objc+1));
    objs[0] = cmdObj;
    memcpy(objs+1, objv, sizeof(Tcl_Obj *) * (unsigned)objc);

    result = Tcl_EvalObjv(interp, objc+1, objs, 0);

    Tcl_DecrRefCount(cmdObj);
    ckfree((char *) objs);

    return result;
}

/*
 *----------------------------------------------------------------------
 *
 * Tk_ChooseColorObjCmd --
 *
 *	This procedure implements the color dialog box for the Unix platform.
 *	See the user documentation for details on what it does.
 *
 * Results:
 *	See user documentation.
 *
 * Side effects:
 *	A dialog window is created the first time this procedure is called.
 *	This window is not destroyed and will be reused the next time the
 *	application invokes the "tk_chooseColor" command.
 *
 *----------------------------------------------------------------------
 */

int
Tk_ChooseColorObjCmd(
    ClientData clientData,	/* Main window associated with interpreter. */
    Tcl_Interp *interp,		/* Current interpreter. */
    int objc,			/* Number of arguments. */
    Tcl_Obj *CONST *objv)	/* Arguments. */
{
    return EvalObjv(interp, "tk::ColorDialog", objc-1, objv+1);
}

/*
 *----------------------------------------------------------------------
 *
 * Tk_GetOpenFileCmd --
 *
 *	This procedure implements the "open file" dialog box for the Unix
 *	platform. See the user documentation for details on what it does.
 *
 * Results:
 *	See user documentation.
 *
 * Side effects:
 *	A dialog window is created the first this procedure is called. This
 *	window is not destroyed and will be reused the next time the
 *	application invokes the "tk_getOpenFile" or "tk_getSaveFile" command.
 *
 *----------------------------------------------------------------------
 */

int
Tk_GetOpenFileObjCmd(
    ClientData clientData,	/* Main window associated with interpreter. */
    Tcl_Interp *interp,		/* Current interpreter. */
    int objc,			/* Number of arguments. */
    Tcl_Obj *CONST *objv)	/* Arguments. */
{
    Tk_Window tkwin = (Tk_Window)clientData;

    if (Tk_StrictMotif(tkwin)) {
	return EvalObjv(interp, "tk::MotifOpenFDialog", objc-1, objv+1);
    } else {
	return EvalObjv(interp, "tk::OpenFDialog", objc-1, objv+1);
    }
}

/*
 *----------------------------------------------------------------------
 *
 * Tk_GetSaveFileCmd --
 *
 *	Same as Tk_GetOpenFileCmd but opens a "save file" dialog box instead.
 *
 * Results:
 *	Same as Tk_GetOpenFileCmd.
 *
 * Side effects:
 *	Same as Tk_GetOpenFileCmd.
 *
 *----------------------------------------------------------------------
 */

int
Tk_GetSaveFileObjCmd(
    ClientData clientData,	/* Main window associated with interpreter. */
    Tcl_Interp *interp,		/* Current interpreter. */
    int objc,			/* Number of arguments. */
    Tcl_Obj *CONST *objv)	/* Arguments. */
{
    Tk_Window tkwin = (Tk_Window)clientData;

    if (Tk_StrictMotif(tkwin)) {
	return EvalObjv(interp, "tk::MotifSaveFDialog", objc-1, objv+1);
    } else {
	return EvalObjv(interp, "tk::SaveFDialog", objc-1, objv+1);
    }
}

/*
 *----------------------------------------------------------------------
 *
 * Tk_MessageBoxCmd --
 *
 *	This procedure implements the MessageBox window for the Unix
 *	platform. See the user documentation for details on what it does.
 *
 * Results:
 *	See user documentation.
 *
 * Side effects:
 *	None. The MessageBox window will be destroy before this procedure
 *	returns.
 *
 *----------------------------------------------------------------------
 */

int
Tk_MessageBoxCmd(
    ClientData clientData,	/* Main window associated with interpreter. */
    Tcl_Interp *interp,		/* Current interpreter. */
    int objc,			/* Number of arguments. */
    Tcl_Obj *CONST *objv)	/* Arguments. */
{
    return EvalObjv(interp, "tk::MessageBox", objc-1, objv+1);
}

#endif /* TK_OBSOLETE_UNIX_DIALOG_WRAPPERS */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
