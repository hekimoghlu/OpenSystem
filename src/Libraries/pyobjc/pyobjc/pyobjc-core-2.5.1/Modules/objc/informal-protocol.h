/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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

#ifndef PyObjC_INFORMAL_PROTOCOL_H
#define PyObjC_INFORMAL_PROTOCOL_H
/*!
 * @header informal-protocol.h
 * @abstruct Support for informal protocols
 * @discussion
 * 	This module defines functions and types for working with informal 
 * 	protocols. 
 *
 * 	NOTE: We also use these functions when looking for the method signatures
 * 	declared in formal protocols, as we don't have specific support for
 * 	formal protocols.
 */

extern PyTypeObject PyObjCInformalProtocol_Type;
#define PyObjCInformalProtocol_Check(obj) PyObject_TypeCheck(obj, &PyObjCInformalProtocol_Type)

int PyObjCInformalProtocol_CheckClass(PyObject*, char*, PyObject*, PyObject*);
PyObject* PyObjCInformalProtocol_FindSelector(PyObject* obj, SEL selector, int isClassMethod);
int PyObjCInformalProtocol_Warnings(char* name, PyObject* clsdict, PyObject* protocols);
PyObject* PyObjCInformalProtocol_FindProtocol(SEL selector);

/* TODO: rename */
PyObject* findSelInDict(PyObject* clsdict, SEL selector);
int signaturesEqual(const char* sig1, const char* sig2);



#endif /* PyObjC_INFORMAL_PROTOCOL_H */
