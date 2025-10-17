/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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
 * Thread extension version numbers are not stored here
 * because this isn't a public export file.
 */

#ifndef _TCL_THREAD_H_
#define _TCL_THREAD_H_

#include <tcl.h>
#include <stdlib.h> /* For strtoul */
#include <string.h> /* For memset and friends */

#undef  TCL_STORAGE_CLASS
#define TCL_STORAGE_CLASS DLLEXPORT

/*
 * For linking against AOLserver require V4 at least
 */

#ifdef NS_AOLSERVER
# include <ns.h>
# if !defined(NS_MAJOR_VERSION) || NS_MAJOR_VERSION < 4
#  error "unsupported AOLserver version"
# endif
#endif

/*
 * Allow for some command names customization.
 * Only thread:: and tpool:: are handled here.
 * Shared variable commands are more complicated.
 * Look into the threadSvCmd.h for more info.
 */

#define THREAD_CMD_PREFIX "thread::"
#define TPOOL_CMD_PREFIX  "tpool::"

/*
 * Exported from threadCmd.c file.
 */

EXTERN int Thread_Init       _ANSI_ARGS_((Tcl_Interp *interp));
EXTERN int Thread_SafeInit   _ANSI_ARGS_((Tcl_Interp *interp));
EXTERN int Thread_Unload     _ANSI_ARGS_((Tcl_Interp *interp));
EXTERN int Thread_SafeUnload _ANSI_ARGS_((Tcl_Interp *interp));

/*
 * Exported from threadSvCmd.c file.
 */

EXTERN int Sv_Init _ANSI_ARGS_((Tcl_Interp *interp));

/*
 * Exported from threadSpCmd.c file.
 */

EXTERN int Sp_Init _ANSI_ARGS_((Tcl_Interp *interp));

/*
 * Exported from threadPoolCmd.c file.
 */

EXTERN int Tpool_Init _ANSI_ARGS_((Tcl_Interp *interp));

/*
 * Macros for splicing in/out of linked lists
 */

#define SpliceIn(a,b)                          \
    (a)->nextPtr = (b);                        \
    if ((b) != NULL)                           \
        (b)->prevPtr = (a);                    \
    (a)->prevPtr = NULL, (b) = (a)

#define SpliceOut(a,b)                         \
    if ((a)->prevPtr != NULL)                  \
        (a)->prevPtr->nextPtr = (a)->nextPtr;  \
    else                                       \
        (b) = (a)->nextPtr;                    \
    if ((a)->nextPtr != NULL)                  \
        (a)->nextPtr->prevPtr = (a)->prevPtr

/*
 * Utility macros
 */ 

#define TCL_CMD(a,b,c) \
  if (Tcl_CreateObjCommand((a),(b),(c),(ClientData)NULL, NULL) == NULL) \
    return TCL_ERROR

#define OPT_CMP(a,b) \
  ((a) && (b) && (*(a)==*(b)) && (*(a+1)==*(b+1)) && (!strcmp((a),(b))))

#ifndef TCL_TSD_INIT
#define TCL_TSD_INIT(keyPtr) \
  (ThreadSpecificData*)Tcl_GetThreadData((keyPtr),sizeof(ThreadSpecificData))
#endif

#undef  TCL_STORAGE_CLASS
#define TCL_STORAGE_CLASS DLLIMPORT

#endif /* _TCL_THREAD_H_ */
