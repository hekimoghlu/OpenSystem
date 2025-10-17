/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 3, 2022.
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
/* 
 * tclMacAppInit.c --
 *
 *	Provides a version of the Tcl_AppInit procedure for the example shell.
 *
 * Copyright (c) 1993-1994 Lockheed Missle & Space Company, AI Center
 * Copyright (c) 1995-1997 Sun Microsystems, Inc.
 *
 * See the file "license.terms" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 *
 * SCCS: @(#) tclMacAppInit.c 1.17 97/01/21 18:13:34
 */

#include "tcl.h"
#include "tclInt.h"
#include "tclMacInt.h"

#if defined(THINK_C)
#   include <console.h>
#elif defined(__MWERKS__)
#   include <SIOUX.h>
short InstallConsole _ANSI_ARGS_((short fd));
#endif



/*
 *----------------------------------------------------------------------
 *
 * MacintoshInit --
 *
 *	This procedure calls initalization routines to set up a simple
 *	console on a Macintosh.  This is necessary as the Mac doesn't
 *	have a stdout & stderr by default.
 *
 * Results:
 *	Returns TCL_OK if everything went fine.  If it didn't the 
 *	application should probably fail.
 *
 * Side effects:
 *	Inits the appropiate console package.
 *
 *----------------------------------------------------------------------
 */

#ifdef __cplusplus
extern "C"
#endif
extern int
MacintoshInit()
{
#if defined(THINK_C)

    /* Set options for Think C console package */
    /* The console package calls the Mac init calls */
    console_options.pause_atexit = 0;
    console_options.title = "\pTcl Interpreter";
		
#elif defined(__MWERKS__)

    /* Set options for CodeWarrior SIOUX package */
    SIOUXSettings.autocloseonquit = true;
    SIOUXSettings.showstatusline = true;
    SIOUXSettings.asktosaveonclose = false;
    InstallConsole(0);
    SIOUXSetTitle("\pTcl Interpreter");
		
#elif defined(applec)

    /* Init packages used by MPW SIOW package */
    InitGraf((Ptr)&qd.thePort);
    InitFonts();
    InitWindows();
    InitMenus();
    TEInit();
    InitDialogs(nil);
    InitCursor();
		
#endif

    TclMacSetEventProc((TclMacConvertEventPtr) SIOUXHandleOneEvent);
    
    /* No problems with initialization */
    return TCL_OK;
}
