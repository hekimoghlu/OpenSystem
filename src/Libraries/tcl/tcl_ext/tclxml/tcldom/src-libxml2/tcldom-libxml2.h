/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 19, 2023.
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
#ifndef __TCLDOM_LIBXML2_H__
#define __TCLDOM_LIBXML2_H__

#include <tcldom.h>
#include <libxml/tree.h>

/*
 * For C++ compilers, use extern "C"
 */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * These macros are used to control whether functions are being declared for
 * import or export in Windows, 
 * They map to no-op declarations on non-Windows systems.
 * Assumes that tcl.h defines DLLEXPORT & DLLIMPORT correctly.
 * The default build on windows is for a DLL, which causes the DLLIMPORT
 * and DLLEXPORT macros to be nonempty. To build a static library, the
 * macro STATIC_BUILD should be defined before the inclusion of tcl.h
 *
 * If a function is being declared while it is being built
 * to be included in a shared library, then it should have the DLLEXPORT
 * storage class.  If is being declared for use by a module that is going to
 * link against the shared library, then it should have the DLLIMPORT storage
 * class.  If the symbol is beind declared for a static build or for use from a
 * stub library, then the storage class should be empty.
 *
 * The convention is that a macro called BUILD_xxxx, where xxxx is the
 * name of a library we are building, is set on the compile line for sources
 * that are to be placed in the library.  When this macro is set, the
 * storage class will be set to DLLEXPORT.  At the end of the header file, the
 * storage class will be reset to DLLIMPORt.
 */

#undef TCL_STORAGE_CLASS
#ifdef BUILD_Tcldomxml
# define TCL_STORAGE_CLASS DLLEXPORT
#else
# ifdef USE_TCL_STUBS
#  define TCL_STORAGE_CLASS
# else
#  define TCL_STORAGE_CLASS DLLIMPORT
# endif
#endif

EXTERN Tcl_FreeInternalRepProc	TclDOM_DocFree;
EXTERN Tcl_DupInternalRepProc	TclDOM_DocDup;
EXTERN Tcl_UpdateStringProc	TclDOM_DocUpdate;
EXTERN Tcl_SetFromAnyProc	TclDOM_DocSetFromAny;

EXTERN Tcl_FreeInternalRepProc	TclDOM_NodeFree;
EXTERN Tcl_DupInternalRepProc	TclDOM_NodeDup;
EXTERN Tcl_UpdateStringProc	TclDOM_NodeUpdate;
EXTERN Tcl_SetFromAnyProc	TclDOM_NodeSetFromAny;

/*
 * Object types
 */

EXTERN Tcl_ObjType TclDOM_DocObjType;
EXTERN Tcl_ObjType TclDOM_NodeObjType;

/*
 * The following function is required to be defined in all stubs aware
 * extensions of TclDOM.  The function is actually implemented in the stub
 * library, not the main Tcldom library, although there is a trivial
 * implementation in the main library in case an extension is statically
 * linked into an application.
 */

EXTERN CONST char *	Tcldomxml_InitStubs _ANSI_ARGS_((Tcl_Interp *interp,
			    CONST char *version, int exact));

#ifndef USE_TCLDOMXML_STUBS

/*
 * When not using stubs, make it a macro.
 */

#define Tcldomxml_InitStubs(interp, version, exact) \
    Tcl_PkgRequire(interp, "dom::generic", version, exact)

#endif

/*
 * Accessor functions => Stubs
 */

#include "tcldomxmlDecls.h"

#undef  TCL_STORAGE_CLASS
#define TCL_STORAGE_CLASS DLLIMPORT

#ifdef __cplusplus
}
#endif

#endif /* TCLDOM_LIBXML2_H__ */
