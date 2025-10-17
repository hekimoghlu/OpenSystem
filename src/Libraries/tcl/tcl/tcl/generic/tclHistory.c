/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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
#include "tclInt.h"

/*
 *----------------------------------------------------------------------
 *
 * Tcl_RecordAndEval --
 *
 *	This procedure adds its command argument to the current list of
 *	recorded events and then executes the command by calling Tcl_Eval.
 *
 * Results:
 *	The return value is a standard Tcl return value, the result of
 *	executing cmd.
 *
 * Side effects:
 *	The command is recorded and executed.
 *
 *----------------------------------------------------------------------
 */

int
Tcl_RecordAndEval(
    Tcl_Interp *interp,		/* Token for interpreter in which command will
				 * be executed. */
    CONST char *cmd,		/* Command to record. */
    int flags)			/* Additional flags. TCL_NO_EVAL means only
				 * record: don't execute command.
				 * TCL_EVAL_GLOBAL means use Tcl_GlobalEval
				 * instead of Tcl_Eval. */
{
    register Tcl_Obj *cmdPtr;
    int length = strlen(cmd);
    int result;

    if (length > 0) {
	/*
	 * Call Tcl_RecordAndEvalObj to do the actual work.
	 */

	cmdPtr = Tcl_NewStringObj(cmd, length);
	Tcl_IncrRefCount(cmdPtr);
	result = Tcl_RecordAndEvalObj(interp, cmdPtr, flags);

	/*
	 * Move the interpreter's object result to the string result, then
	 * reset the object result.
	 */

	(void) Tcl_GetStringResult(interp);

	/*
	 * Discard the Tcl object created to hold the command.
	 */

	Tcl_DecrRefCount(cmdPtr);
    } else {
	/*
	 * An empty string. Just reset the interpreter's result.
	 */

	Tcl_ResetResult(interp);
	result = TCL_OK;
    }
    return result;
}

/*
 *----------------------------------------------------------------------
 *
 * Tcl_RecordAndEvalObj --
 *
 *	This procedure adds the command held in its argument object to the
 *	current list of recorded events and then executes the command by
 *	calling Tcl_EvalObj.
 *
 * Results:
 *	The return value is a standard Tcl return value, the result of
 *	executing the command.
 *
 * Side effects:
 *	The command is recorded and executed.
 *
 *----------------------------------------------------------------------
 */

int
Tcl_RecordAndEvalObj(
    Tcl_Interp *interp,		/* Token for interpreter in which command will
				 * be executed. */
    Tcl_Obj *cmdPtr,		/* Points to object holding the command to
				 * record and execute. */
    int flags)			/* Additional flags. TCL_NO_EVAL means record
				 * only: don't execute the command.
				 * TCL_EVAL_GLOBAL means evaluate the script
				 * in global variable context instead of the
				 * current procedure. */
{
    int result, call = 1;
    Tcl_Obj *list[3];
    register Tcl_Obj *objPtr;
    Tcl_CmdInfo info;

    /*
     * Do not call [history] if it has been replaced by an empty proc
     */

    result = Tcl_GetCommandInfo(interp, "history", &info);

    if (result && (info.objProc == TclObjInterpProc)) {
	Proc *procPtr = (Proc *)(info.objClientData);
	call = (procPtr->cmdPtr->compileProc != TclCompileNoOp);
    }

    if (call) {

	/*
	 * Do recording by eval'ing a tcl history command: history add $cmd. 
	 */

	TclNewLiteralStringObj(list[0], "history");
	TclNewLiteralStringObj(list[1], "add");
	list[2] = cmdPtr;
	
	objPtr = Tcl_NewListObj(3, list);
	Tcl_IncrRefCount(objPtr);
	(void) Tcl_EvalObjEx(interp, objPtr, TCL_EVAL_GLOBAL);
	Tcl_DecrRefCount(objPtr);
	
	/*
	 * One possible failure mode above: exceeding a resource limit.
	 */
	
	if (Tcl_LimitExceeded(interp)) {
	    return TCL_ERROR;
	}
    }

    /*
     * Execute the command.
     */

    result = TCL_OK;
    if (!(flags & TCL_NO_EVAL)) {
	result = Tcl_EvalObjEx(interp, cmdPtr, flags & TCL_EVAL_GLOBAL);
    }
    return result;
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
