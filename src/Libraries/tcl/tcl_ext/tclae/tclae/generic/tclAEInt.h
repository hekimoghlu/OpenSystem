/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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
#pragma once

#ifndef _TCLAE_INT
#define _TCLAE_INT

#if TARGET_API_MAC_OSX
#define TCLAE_NO_EPPC
#endif

#ifdef TCLAE_USE_FRAMEWORK_INCLUDES
#include <Carbon/Carbon.h>
#else
#include <AppleEvents.h>
#if !defined(TCLAE_NO_EPPC) || defined(TCLAE_CW)
#include <EPPC.h>
#endif
#endif

#include "tclAE.h"

typedef struct cmd_return {
	int     status;
	Tcl_Obj *object;
} CmdReturn;

/* global storing the macRoman Tcl_Encoding */
extern Tcl_Encoding tclAE_macRoman_encoding;

/* Build commands */

int         Tclae_BuildCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_SendCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);

/* Handler commands */

int			Tclae_GetCoercionHandlerCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int			Tclae_GetEventHandlerCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int			Tclae_InstallCoercionHandlerCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int			Tclae_InstallEventHandlerCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int			Tclae_RemoveCoercionHandlerCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int			Tclae_RemoveEventHandlerCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);

OSErr		TclaeErrorCodeFromInterp(Tcl_Interp *interp);
int			TclaeInitEventHandlers(Tcl_Interp *interp);
void		TclaeInitCoercionHandlers(Tcl_Interp *interp);
AEReturnID	TclaeRegisterQueueHandler(Tcl_Interp *interp, Tcl_Obj *replyHandlerProc);

/* Address commands */

int         Tclae_LaunchCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_ProcessesCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_RemoteProcessResolverGetProcessesCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_IPCListPortsCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_PPCBrowserCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int			Tclae_GetAEAddressDescFromObj(Tcl_Interp *interp, Tcl_Obj *objPtr, AEAddressDesc **addressDescPtr);
#if TARGET_API_MAC_CARBON	
int         Tclae_GetPOSIXPathCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_GetHFSPathCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
#endif // TARGET_API_MAC_CARBON

void            TclaeInitAEAddresses();
CmdReturn*      TclaeParse( Tcl_Interp *interp, register AEDesc *desc, Tcl_Obj *parentPtr );

/* AEDesc objects */

void		TclaeInitAEDescs();

Tcl_Obj *	Tclae_NewAEDescObj(AEDesc *descPtr);
Tcl_Obj *	Tclae_NewConstAEDescObj(const AEDesc *descPtr);
Tcl_Obj *	Tclae_NewAEDescRefObj(AEDesc *descPtr);
Tcl_Obj *	Tclae_NewConstAEDescRefObj(const AEDesc *descPtr);
int			Tclae_GetAEDescObjFromObj(Tcl_Interp *interp, Tcl_Obj *objPtr, Tcl_Obj **descObjHdl, int parseGizmo);
int			Tclae_GetAEDescFromObj(Tcl_Interp *interp, Tcl_Obj *objPtr,	AEDesc **descPtr, int parseGizmo);
int			Tclae_GetConstAEDescFromObj(Tcl_Interp *interp,	Tcl_Obj *objPtr, const AEDesc **descPtr, int parseGizmo);
int		Tclae_ConvertToAEDesc(Tcl_Interp *interp, Tcl_Obj *objPtr);

void	TclaeUpdateStringOfAEDesc(Tcl_Obj *objPtr);

void		TclaeDetachAEDescObj(Tcl_Obj *obj);

/* AEDesc commands */

int         Tclae_CoerceDataCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_CoerceDescCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_CountItemsCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_CreateDescCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_CreateListCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_DeleteItemCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_DeleteKeyDescCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_DuplicateDescCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_GetAttributeDataCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_GetAttributeDescCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_GetDataCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int			Tclae_GetDescTypeCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_GetKeyDataCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_GetKeyDescCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_GetNthDataCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_GetNthDescCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_PutDataCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_PutDescCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_PutKeyDataCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_PutKeyDescCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_ReplaceDescDataCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_SetDescTypeCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);

int         Tclae__GetAEDescCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);


CmdReturn*  TclaeCoerceDesc(Tcl_Interp *interp, AEDesc *fromAEDesc, Tcl_Obj *toTypeObjPtr);
void*		TclaeDataFromObj(Tcl_Interp* interp, OSType typeCode, Tcl_Obj* dataObjPtr, Size* dataSizePtr);
int			TclaeGetAttributeDesc(Tcl_Interp *interp, Tcl_Obj *theAppleEventObjPtr, Tcl_Obj *theAttributeObjPtr, Tcl_Obj *theDesiredTypeObjPtr, AEDesc *keyAEDescPtr);
int         TclaeGetKeyDesc(Tcl_Interp *interp, Tcl_Obj *theAERecordObjPtr, Tcl_Obj *theAEKeywordObjPtr, Tcl_Obj *theDesiredTypeObjPtr, AEDesc *keyAEDescPtr);
int         TclaeGetNthDesc(Tcl_Interp *interp, Tcl_Obj *theAEDescListObjPtr, Tcl_Obj *theIndexObjPtr, Tcl_Obj *theDesiredTypeObjPtr, Tcl_Obj *theKeywordVarObjPtr,	AEDesc *nthAEDescPtr);
OSType      TclaeGetOSTypeFromObj(Tcl_Obj *objPtr);
Tcl_Obj*    TclaeNewOSTypeObj(OSType theOSType);
int         TclaePrint(Tcl_Interp *interp, AEDesc *inAEDescPtr);
int         TclaePutDesc(Tcl_Interp *interp, Tcl_Obj *theAEDescListObjPtr, Tcl_Obj *theIndexObjPtr, AEDesc *nthAEDescPtr);
int         TclaePutKeyDesc(Tcl_Interp *interp, Tcl_Obj *theAERecordObjPtr, Tcl_Obj *theAEKeywordObjPtr, AEDesc *keyAEDescPtr);
CmdReturn * TclaeDataFromAEDesc(Tcl_Interp *interp, const AEDesc *theAEDescPtr, Tcl_Obj *desiredTypePtr, Tcl_Obj *typeCodeVarPtr);

Size		TclaeGetDescDataSize(const AEDesc * theAEDesc);
OSErr		TclaeGetDescData(const AEDesc *theAEDesc, void *dataPtr, Size maximumSize);
void *		TclaeAllocateAndGetDescData(const AEDesc *theAEDesc, Size *sizePtr);

/* Object commands */

int         Tclae_SetObjectCallbacksCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_ResolveCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_CallObjectAccessorCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_GetObjectAccessorCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_InstallObjectAccessorCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_RemoveObjectAccessorCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);
int         Tclae_DisposeTokenCmd(ClientData clientData, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[]);

void		TclaeInitObjectAccessors(Tcl_Interp *interp);

#endif /* _TCLAE_INT */
