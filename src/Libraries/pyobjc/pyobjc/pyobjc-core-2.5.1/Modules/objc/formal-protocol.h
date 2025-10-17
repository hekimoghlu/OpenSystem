/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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

#ifndef PyObjC_FORMAL_PROTOCOL_H
#define PyObjC_FORMAL_PROTOCOL_H
/*!
 * @header formal-protocol.h
 * @abstruct Support for formal protocols (aka @protocol)
 * @discussion
 * 	This module defines functions and types for working with formal 
 * 	protocols. 
 *
 * 	NOTE: We also use these functions when looking for the method signatures
 * 	declared in formal protocols, as we don't have specific support for
 * 	formal protocols.
 */

extern PyTypeObject PyObjCFormalProtocol_Type;
#define PyObjCFormalProtocol_Check(obj) PyObject_TypeCheck(obj, &PyObjCFormalProtocol_Type)

int PyObjCFormalProtocol_CheckClass(PyObject*, char*, PyObject*, PyObject*, PyObject*);
const char* PyObjCFormalProtocol_FindSelectorSignature(PyObject* obj, SEL selector, int isClassMethod);
PyObject* PyObjCFormalProtocol_ForProtocol(Protocol* protocol);
Protocol* PyObjCFormalProtocol_GetProtocol(PyObject* protocol);

#endif /* PyObjC_FORMAL_PROTOCOL_H */
