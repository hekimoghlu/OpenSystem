/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 27, 2025.
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
/*#include <stdlib.h>*/
#include "memchanInt.h"
#include "buf.h"

extern BufStubs bufStubs;

char *
Buf_InitStubs _ANSI_ARGS_((Tcl_Interp *interp, CONST char *version, int exact));

#ifndef PACKAGE_VERSION
#define PACKAGE_VERSION MEMCHAN_VERSION
#endif
#ifndef PACKAGE_NAME
#define PACKAGE_NAME "Memchan"
#endif


/*
 *------------------------------------------------------*
 *
 *	Memchan_Init --
 *
 *	------------------------------------------------*
 *	Standard procedure required by 'load'. 
 *	Initializes this extension.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		As of 'Tcl_CreateCommand'.
 *
 *	Result:
 *		A standard Tcl error code.
 *
 *------------------------------------------------------*
 */

int Memchan_Init (interp)
Tcl_Interp* interp;
{
#if GT81
  if (Tcl_InitStubs (interp, "8.1", 0) == NULL) {
    return TCL_ERROR;
  }
#endif

  Tcl_CreateObjCommand (interp, "memchan",
			&MemchanCmd,
			(ClientData) NULL,
			(Tcl_CmdDeleteProc*) NULL);

  Tcl_CreateObjCommand (interp, "fifo",
			&MemchanFifoCmd,
			(ClientData) NULL,
			(Tcl_CmdDeleteProc*) NULL);

  Tcl_CreateObjCommand (interp, "fifo2",
			&MemchanFifo2Cmd,
			(ClientData) NULL,
			(Tcl_CmdDeleteProc*) NULL);

  Tcl_CreateObjCommand (interp, "null",
			&MemchanNullCmd,
			(ClientData) NULL,
			(Tcl_CmdDeleteProc*) NULL);

  Tcl_CreateObjCommand (interp, "random",
			&MemchanRandomCmd,
			(ClientData) NULL,
			(Tcl_CmdDeleteProc*) NULL);

  Tcl_CreateObjCommand (interp, "zero",
			&MemchanZeroCmd,
			(ClientData) NULL,
			(Tcl_CmdDeleteProc*) NULL);

#if GT81
    /* register extension and its interfaces as now available package
     */
    Tcl_PkgProvideEx (interp, PACKAGE_NAME, PACKAGE_VERSION, (ClientData) &bufStubs);

#ifndef __WIN32__
    Buf_InitStubs (interp, PACKAGE_VERSION, 0);
#endif
#else
  /* register memory channels as available package */
  Tcl_PkgProvide (interp, PACKAGE_NAME, PACKAGE_VERSION);
#endif

  Buf_Init (interp);
  return TCL_OK;
}

/*
 *------------------------------------------------------*
 *
 *	Memchan_SafeInit --
 *
 *	------------------------------------------------*
 *	Standard procedure required by 'load'. 
 *	Initializes this extension for a safe interpreter.
 *	------------------------------------------------*
 *
 *	Sideeffects:
 *		As of 'Memchan_Init'
 *
 *	Result:
 *		A standard Tcl error code.
 *
 *------------------------------------------------------*
 */

int Memchan_SafeInit (interp)
Tcl_Interp* interp;
{
  return Memchan_Init (interp);
}

