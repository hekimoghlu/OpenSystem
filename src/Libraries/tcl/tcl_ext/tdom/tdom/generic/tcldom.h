/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
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
#ifndef __TCLDOM_H_INCLUDE__
#define __TCLDOM_H_INCLUDE__  

#include <tcl.h>


/* The following procs are defined in tcldom.c - since they are used
 * in nodecmd.c these need a prototype somewhere. The prototypes can
 * live here for now. */
int  tcldom_textCheck(Tcl_Interp *interp, char *text, char *errText);
int  tcldom_commentCheck(Tcl_Interp *interp, char *text);
int  tcldom_CDATACheck(Tcl_Interp *interp, char *text);
int  tcldom_PIValueCheck(Tcl_Interp *interp, char *text);
int  tcldom_PINameCheck(Tcl_Interp *interp, char *name);
int  tcldom_nameCheck(Tcl_Interp *interp, char *name, char *nameType,
                      int isFQName);
void tcldom_createNodeObj(Tcl_Interp * interp, domNode *node,
                          char *objCmdName);


void tcldom_initialize(void);

Tcl_ObjCmdProc tcldom_DomObjCmd;
Tcl_ObjCmdProc tcldom_DocObjCmd;
Tcl_ObjCmdProc tcldom_NodeObjCmd;
Tcl_ObjCmdProc TclExpatObjCmd;
Tcl_ObjCmdProc tcldom_unknownCmd;
Tcl_ObjCmdProc TclTdomObjCmd;

#if defined(_MSC_VER)
#  undef TCL_STORAGE_CLASS
#  define TCL_STORAGE_CLASS DLLEXPORT
#endif

#define STR_TDOM_VERSION(v) (VERSION)

EXTERN int Tdom_Init     _ANSI_ARGS_((Tcl_Interp *interp));
EXTERN int Tdom_SafeInit _ANSI_ARGS_((Tcl_Interp *interp));

#endif


