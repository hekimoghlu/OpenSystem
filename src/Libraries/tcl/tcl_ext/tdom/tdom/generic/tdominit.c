/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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
/*----------------------------------------------------------------------------
|   Includes
|
\---------------------------------------------------------------------------*/
#include <tcl.h>
#include <dom.h>
#include <tdom.h>
#include <tcldom.h>

extern TdomStubs tdomStubs;

/*
 *----------------------------------------------------------------------------
 *
 * Tdom_Init --
 *
 *	Initialization routine for loadable module
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Defines "expat"/"dom" commands in the interpreter.
 *
 *----------------------------------------------------------------------------
 */

int
Tdom_Init (interp)
     Tcl_Interp *interp; /* Interpreter to initialize. */
{

#ifdef USE_TCL_STUBS
    Tcl_InitStubs(interp, "8", 0);
#endif

    domModuleInitialize();

#ifdef TCL_THREADS
    tcldom_initialize();
#endif /* TCL_THREADS */

#ifndef TDOM_NO_UNKNOWN_CMD
    Tcl_Eval(interp, "rename unknown unknown_tdom");   
    Tcl_CreateObjCommand(interp, "unknown", tcldom_unknownCmd,  NULL, NULL );
#endif

    Tcl_CreateObjCommand(interp, "dom",     tcldom_DomObjCmd,   NULL, NULL );
    Tcl_CreateObjCommand(interp, "domDoc",  tcldom_DocObjCmd,   NULL, NULL );
    Tcl_CreateObjCommand(interp, "domNode", tcldom_NodeObjCmd,  NULL, NULL );
    Tcl_CreateObjCommand(interp, "tdom",    TclTdomObjCmd,      NULL, NULL );

#ifndef TDOM_NO_EXPAT    
    Tcl_CreateObjCommand(interp, "expat",       TclExpatObjCmd, NULL, NULL );
    Tcl_CreateObjCommand(interp, "xml::parser", TclExpatObjCmd, NULL, NULL );
#endif
    
#ifdef USE_TCL_STUBS
    Tcl_PkgProvideEx(interp, PACKAGE_NAME, PACKAGE_VERSION, 
                     (ClientData) &tdomStubs);
#else
    Tcl_PkgProvide(interp, PACKAGE_NAME, PACKAGE_VERSION);
#endif

    return TCL_OK;
}

int
Tdom_SafeInit (interp)
     Tcl_Interp *interp;
{
    return Tdom_Init (interp);
}

/*
 * Load the AOLserver stub. This allows the library
 * to be loaded as AOLserver module.
 */

#if defined (NS_AOLSERVER)
# include "aolstub.cpp"
#endif

