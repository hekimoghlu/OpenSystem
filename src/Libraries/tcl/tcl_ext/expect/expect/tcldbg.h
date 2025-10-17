/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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
/* _DEBUG or _DBG is just too likely, use something more unique */
#ifndef _NIST_DBG
#define _NIST_DBG

#include "tcl.h"

typedef int (Dbg_InterProc) _ANSI_ARGS_((Tcl_Interp *interp, ClientData data));
typedef int (Dbg_IgnoreFuncsProc) _ANSI_ARGS_((
			Tcl_Interp *interp,
			char *funcname));
typedef void (Dbg_OutputProc) _ANSI_ARGS_((
			Tcl_Interp *interp,
			char *output,
			ClientData data));

typedef struct {
  Dbg_InterProc *func;
  ClientData data;
} Dbg_InterStruct;

typedef struct {
  Dbg_OutputProc *func;
  ClientData data;
} Dbg_OutputStruct;

EXTERN char *Dbg_VarName;
EXTERN char *Dbg_DefaultCmdName;

/* trivial interface, creates a "debug" command in your interp */
EXTERN int Tcldbg_Init _ANSI_ARGS_((Tcl_Interp *));

EXTERN void Dbg_On _ANSI_ARGS_((Tcl_Interp *interp,
					int immediate));
EXTERN void Dbg_Off _ANSI_ARGS_((Tcl_Interp *interp));
EXTERN char **Dbg_ArgcArgv _ANSI_ARGS_((int argc,char *argv[],
					int copy));
EXTERN int Dbg_Active _ANSI_ARGS_((Tcl_Interp *interp));
EXTERN Dbg_InterStruct Dbg_Interactor _ANSI_ARGS_((
					Tcl_Interp *interp,
					Dbg_InterProc *interactor,
					ClientData data));
EXTERN Dbg_IgnoreFuncsProc *Dbg_IgnoreFuncs _ANSI_ARGS_((
					Tcl_Interp *interp,
					Dbg_IgnoreFuncsProc *));
EXTERN Dbg_OutputStruct Dbg_Output _ANSI_ARGS_((
					Tcl_Interp *interp,
					Dbg_OutputProc *,
					ClientData data));

EXTERN void Dbg_StdinMode _ANSI_ARGS_((int mode));

#endif /* _NIST_DBG */
