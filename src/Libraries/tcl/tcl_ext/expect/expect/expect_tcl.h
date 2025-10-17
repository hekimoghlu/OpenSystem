/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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
#ifndef _EXPECT_TCL_H
#define _EXPECT_TCL_H

#include <stdio.h>
#include "expect_comm.h"

/*
 * This is a convenience macro used to initialize a thread local storage ptr.
 * Stolen from tclInt.h
 */
#ifndef TCL_TSD_INIT
#define TCL_TSD_INIT(keyPtr)	(ThreadSpecificData *)Tcl_GetThreadData((keyPtr), sizeof(ThreadSpecificData))
#endif

EXTERN int exp_cmdlinecmds;
EXTERN int exp_interactive;
EXTERN FILE *exp_cmdfile;
EXTERN char *exp_cmdfilename;
EXTERN int exp_getpid;	/* pid of Expect itself */
EXTERN int exp_buffer_command_input;

EXTERN int exp_strict_write;

EXTERN int exp_tcl_debugger_available;

EXTERN Tcl_Interp *exp_interp;

#define Exp_Init Expect_Init
EXTERN int	Expect_Init _ANSI_ARGS_((Tcl_Interp *));	/* for Tcl_AppInit apps */
EXTERN void	exp_parse_argv _ANSI_ARGS_((Tcl_Interp *,int argc,char **argv));
EXTERN int	exp_interpreter _ANSI_ARGS_((Tcl_Interp *,Tcl_Obj *));
EXTERN int	exp_interpret_cmdfile _ANSI_ARGS_((Tcl_Interp *,FILE *));
EXTERN int	exp_interpret_cmdfilename _ANSI_ARGS_((Tcl_Interp *,char *));
EXTERN void	exp_interpret_rcfiles _ANSI_ARGS_((Tcl_Interp *,int my_rc,int sys_rc));

EXTERN char *	exp_cook _ANSI_ARGS_((char *s,int *len));

EXTERN void	expCloseOnExec _ANSI_ARGS_((int));

			/* app-specific exit handler */
EXTERN void	(*exp_app_exit)_ANSI_ARGS_((Tcl_Interp *));
EXTERN void	exp_exit_handlers _ANSI_ARGS_((ClientData));

EXTERN void	exp_error _ANSI_ARGS_(TCL_VARARGS(Tcl_Interp *,interp));

#endif /* _EXPECT_TCL_H */
