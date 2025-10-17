/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 19, 2023.
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
#ifndef _PROTBCLOAD_H
# define _PROTBCLOAD_H

# include "tcl.h"

# ifdef BUILD_tbcload
# undef TCL_STORAGE_CLASS
# define TCL_STORAGE_CLASS DLLEXPORT
# endif

/*
 *----------------------------------------------------------------
 * Procedures exported by cmpRead.c and cmpRPkg.c
 *----------------------------------------------------------------
 */

EXTERN int	Tbcload_EvalObjCmd _ANSI_ARGS_((ClientData dummy,
			Tcl_Interp *interp, int objc, Tcl_Obj *CONST objv[]));
EXTERN int	Tbcload_ProcObjCmd _ANSI_ARGS_((ClientData dummy,
			Tcl_Interp *interp, int objc, Tcl_Obj *CONST objv[]));
EXTERN CONST char *
		TbcloadGetPackageName _ANSI_ARGS_((void));

EXTERN int	Tbcload_Init _ANSI_ARGS_((Tcl_Interp *interp));
EXTERN int	Tbcload_SafeInit _ANSI_ARGS_((Tcl_Interp *interp));

# undef TCL_STORAGE_CLASS
# define TCL_STORAGE_CLASS DLLIMPORT

#endif /* _PROTBCLOAD_H */
