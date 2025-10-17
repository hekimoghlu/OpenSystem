/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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
#ifndef INCLUDED_QEBIND_H
#define INCLUDED_QEBIND_H

/*
 * Used to tag functions that are only to be visible within the module being
 * built and not outside it (where this is supported by the linker).
 */

#ifndef MODULE_SCOPE
#   ifdef __cplusplus
#	define MODULE_SCOPE extern "C"
#   else
#	define MODULE_SCOPE extern
#   endif
#endif

typedef struct QE_BindingTable_ *QE_BindingTable;

/* Pass to QE_BindEvent */
typedef struct QE_Event {
	int type;
	int detail;
	ClientData clientData;
} QE_Event;

typedef struct QE_ExpandArgs {
	QE_BindingTable bindingTable;
	char which;
	ClientData object;
	Tcl_DString *result;
	int event;
	int detail;
	ClientData clientData;
} QE_ExpandArgs;

typedef void (*QE_ExpandProc)(QE_ExpandArgs *args);

MODULE_SCOPE int debug_bindings;

MODULE_SCOPE int QE_BindInit(Tcl_Interp *interp);
MODULE_SCOPE QE_BindingTable QE_CreateBindingTable(Tcl_Interp *interp);
MODULE_SCOPE void QE_DeleteBindingTable(QE_BindingTable bindingTable);
MODULE_SCOPE int QE_InstallEvent(QE_BindingTable bindingTable, char *name, QE_ExpandProc expand);
MODULE_SCOPE int QE_InstallDetail(QE_BindingTable bindingTable, char *name, int eventType, QE_ExpandProc expand);
MODULE_SCOPE int QE_UninstallEvent(QE_BindingTable bindingTable, int eventType);
MODULE_SCOPE int QE_UninstallDetail(QE_BindingTable bindingTable, int eventType, int detail);
MODULE_SCOPE int QE_CreateBinding(QE_BindingTable bindingTable,
	ClientData object, char *eventString, char *command, int append);
MODULE_SCOPE int QE_DeleteBinding(QE_BindingTable bindingTable,
	ClientData object, char *eventString);
MODULE_SCOPE int QE_GetAllObjects(QE_BindingTable bindingTable);
MODULE_SCOPE int QE_GetBinding(QE_BindingTable bindingTable,
	ClientData object, char *eventString);
MODULE_SCOPE int QE_GetAllBindings(QE_BindingTable bindingTable,
	ClientData object);
MODULE_SCOPE int QE_GetEventNames(QE_BindingTable bindingTable);
MODULE_SCOPE int QE_GetDetailNames(QE_BindingTable bindingTable, char *eventName);
MODULE_SCOPE int QE_BindEvent(QE_BindingTable bindingTable, QE_Event *eventPtr);
MODULE_SCOPE void QE_ExpandDouble(double number, Tcl_DString *result);
MODULE_SCOPE void QE_ExpandNumber(long number, Tcl_DString *result);
MODULE_SCOPE void QE_ExpandString(char *string, Tcl_DString *result);
MODULE_SCOPE void QE_ExpandEvent(QE_BindingTable bindingTable, int eventType, Tcl_DString *result);
MODULE_SCOPE void QE_ExpandDetail(QE_BindingTable bindingTable, int event, int detail, Tcl_DString *result);
MODULE_SCOPE void QE_ExpandPattern(QE_BindingTable bindingTable, int eventType, int detail, Tcl_DString *result);
MODULE_SCOPE void QE_ExpandUnknown(char which, Tcl_DString *result);
MODULE_SCOPE int QE_BindCmd(QE_BindingTable bindingTable, int objOffset, int objc,
	Tcl_Obj *CONST objv[]);
MODULE_SCOPE int QE_ConfigureCmd(QE_BindingTable bindingTable, int objOffset, int objc,
	Tcl_Obj *CONST objv[]);
MODULE_SCOPE int QE_GenerateCmd(QE_BindingTable bindingTable, int objOffset, int objc,
	Tcl_Obj *CONST objv[]);
MODULE_SCOPE int QE_InstallCmd(QE_BindingTable bindingTable, int objOffset, int objc,
	Tcl_Obj *CONST objv[]);
MODULE_SCOPE int QE_UnbindCmd(QE_BindingTable bindingTable, int objOffset, int objc,
	Tcl_Obj *CONST objv[]);
MODULE_SCOPE int QE_UninstallCmd(QE_BindingTable bindingTable, int objOffset, int objc,
	Tcl_Obj *CONST objv[]);
MODULE_SCOPE int QE_LinkageCmd(QE_BindingTable bindingTable, int objOffset, int objc,
	Tcl_Obj *CONST objv[]);

#endif /* INCLUDED_QEBIND_H */

