/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
#include "xotclInt.h"

int
XOTclErrMsg(Tcl_Interp *interp, char *msg, Tcl_FreeProc* type) {
    Tcl_SetResult(interp, msg, type);
    return TCL_ERROR;
}

int
XOTclVarErrMsg TCL_VARARGS_DEF (Tcl_Interp *, arg1)
{
    va_list argList;
    char *string;
    Tcl_Interp *interp;

    interp = TCL_VARARGS_START(Tcl_Interp *, arg1, argList);
    Tcl_ResetResult(interp);
    while (1) {
      string = va_arg(argList, char *);
      if (string == NULL) {
        break;
      }
      Tcl_AppendResult(interp, string, (char *) NULL);
    }
    va_end(argList);
    return TCL_ERROR;
}


int
XOTclErrInProc (Tcl_Interp *interp, Tcl_Obj *objName,
		Tcl_Obj *clName, char *procName) {
    Tcl_DString errMsg;
    char *cName, *space;
    ALLOC_DSTRING(&errMsg, "\n    ");
    if (clName) {
      cName = ObjStr(clName);
      space = " ";
    } else {
      cName = "";
      space ="";
    }
    Tcl_DStringAppend(&errMsg, ObjStr(objName),-1);
    Tcl_DStringAppend(&errMsg, space, -1);
    Tcl_DStringAppend(&errMsg, cName, -1);
    Tcl_DStringAppend(&errMsg, "->", 2);
    Tcl_DStringAppend(&errMsg, procName, -1);
    Tcl_AddErrorInfo (interp, Tcl_DStringValue(&errMsg));
    DSTRING_FREE(&errMsg);
    return TCL_ERROR;
}

int
XOTclObjErrArgCnt(Tcl_Interp *interp, Tcl_Obj *cmdname, char *arglist) {
  Tcl_ResetResult(interp);
  Tcl_AppendResult(interp, "wrong # args: should be {", (char *) NULL);
  if (cmdname) {
    Tcl_AppendResult(interp, ObjStr(cmdname), " ", (char *) NULL);
  }
  if (arglist != 0) Tcl_AppendResult(interp, arglist, (char *) NULL);
  Tcl_AppendResult(interp, "}", (char *) NULL);
  return TCL_ERROR;
}

int
XOTclErrBadVal(Tcl_Interp *interp, char *context, char *expected, char *value) {
  Tcl_ResetResult(interp);
  Tcl_AppendResult(interp, context, ": expected ", expected, " but got '", 
		   value, "'", (char *) NULL);
  return TCL_ERROR;
}

int
XOTclErrBadVal_(Tcl_Interp *interp, char *expected, char *value) {
  fprintf(stderr, "Deprecated call, recompile your program with xotcl 1.5 or newer\n");
  Tcl_ResetResult(interp);
  Tcl_AppendResult(interp, ": expected ", expected, " but got '", 
		   value, "'", (char *) NULL);
  return TCL_ERROR;
}

extern int
XOTclObjErrType(Tcl_Interp *interp, Tcl_Obj *nm, char *wt) {
  Tcl_ResetResult(interp);
  Tcl_AppendResult(interp,"'",ObjStr(nm), "' method should be called on '",
		   wt, "'", (char *) NULL);
  return TCL_ERROR;
}
