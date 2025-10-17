/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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
#ifndef OSXMACTCL_H
#define OSXMACTCL_H
#pragma once

#ifdef __cplusplus
extern "C" {
#endif
	
#if TARGET_RT_MAC_MACHO

OSErr FSpLocationFromPath (int length, CONST84 char *path, FSRefPtr fileRefPtr);

OSErr FSpPathFromLocation (FSRefPtr fsrefP, int* length, Handle *fullPath);

/* CFString to external DString */
int CFStringToExternalDString(Tcl_Interp * interp, CFStringRef strRef, Tcl_DString * dsPtr);

/* CFString to DString */
int CFStringToUtfDString(Tcl_Interp * interp, CFStringRef strRef, Tcl_DString * dsPtr);

/* decomposed utf8 buffer to external DString */
int DUtfToExternalDString(Tcl_Interp * interp, CONST84 char * src, int length, Tcl_DString * dsPtr);

/* decomposed utf8 buffer to DString */
int DUtfToUtfDString(Tcl_Interp * interp, CONST84 char * src, int length, Tcl_DString * dsPtr);

/* external buffer to decomposed utf8 DString */
int ExternalToDUtfDString(Tcl_Interp * interp, CONST84 char * src, int length, Tcl_DString * dsPtr);

/* utf8 buffer to decomposed utf8 DString */
int UtfToDUtfDString(Tcl_Interp * interp, CONST84 char * src, int length, Tcl_DString * dsPtr);

/* CFStringRef to decomposed Unicode Tcl_Obj */
Tcl_Obj * CFStringToTclObj(CFStringRef strRef);

/* Unicode Tcl_Obj * to CFStringRef */
CFStringRef TclObjToCFString(Tcl_Obj * inObj);

#endif

#ifdef __cplusplus
}
#endif

#endif
