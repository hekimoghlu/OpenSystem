/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 6, 2024.
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
#ifndef ITK_H
#define ITK_H

#ifndef TCL_ALPHA_RELEASE
#   define TCL_ALPHA_RELEASE	0
#endif
#ifndef TCL_BETA_RELEASE
#   define TCL_BETA_RELEASE	1
#endif
#ifndef TCL_FINAL_RELEASE
#   define TCL_FINAL_RELEASE	2
#endif


#define ITK_MAJOR_VERSION	3
#define ITK_MINOR_VERSION	4
#define ITK_RELEASE_LEVEL	TCL_FINAL_RELEASE
#define ITK_RELEASE_SERIAL	0

#define ITK_VERSION		"3.4"
#define ITK_PATCH_LEVEL		"3.4.0"


/*
 * A special definition used to allow this header file to be included
 * in resource files so that they can get obtain version information from
 * this file.  Resource compilers don't like all the C stuff, like typedefs
 * and procedure declarations, that occur below.
 */

#ifndef RC_INVOKED

#include "tk.h"
#include "itclInt.h"

#undef TCL_STORAGE_CLASS
#ifdef BUILD_itk
#   define TCL_STORAGE_CLASS DLLEXPORT
#else
#   ifdef USE_ITK_STUBS
#	define TCL_STORAGE_CLASS
#   else
#	define TCL_STORAGE_CLASS DLLIMPORT
#   endif
#endif

/*
 *  List of options in alphabetical order:
 */
typedef struct ItkOptList {
    Tcl_HashTable *options;     /* list containing the real options */
    Tcl_HashEntry **list;       /* gives ordering of options */
    int len;                    /* number of entries in order list */
    int max;                    /* maximum size of order list */
} ItkOptList;

/*
 *  List of options created in the class definition:
 */
typedef struct ItkClassOptTable {
    Tcl_HashTable options;        /* option storage with fast lookup */
    ItkOptList order;             /* gives ordering of options */
} ItkClassOptTable;

/*
 *  Each option created in the class definition:
 */
typedef struct ItkClassOption {
    ItclMember *member;           /* info about this option */
    char *resName;                /* resource name in X11 database */
    char *resClass;               /* resource class name in X11 database */
    char *init;                   /* initial value for option */
} ItkClassOption;

#include "itkDecls.h"

/*
 *  This function is contained in the itkstub static library
 */

#ifdef USE_ITK_STUBS
TCL_EXTERNC CONST char *
	Itk_InitStubs _ANSI_ARGS_((Tcl_Interp *interp,
			    CONST char *version, int exact));
#else
#define Itk_InitStubs(interp, version, exact) \
      Tcl_PkgRequire(interp, "Itk", version, exact)
#endif

/*
 * Public functions that are not accessible via the stubs table.
 */

#undef TCL_STORAGE_CLASS
#define TCL_STORAGE_CLASS DLLIMPORT

#endif /* RC_INVOKED */
#endif /* ITK_H */
