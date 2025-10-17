/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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

#ifndef PyObjC_FUNCTION_H
#define PyObjC_FUNCTION_H

extern PyTypeObject PyObjCFunc_Type;

#define PyObjCFunction_Check(value) \
	PyObject_TypeCheck(value, &PyObjCFunc_Type)

extern PyObject*
PyObjCFunc_New(PyObject* name, void* func, const char* signature, PyObject* doc, PyObject* meta);

extern PyObject* 
PyObjCFunc_WithMethodSignature(PyObject* name, void* func, PyObjCMethodSignature* signature);

extern PyObjCMethodSignature*
PyObjCFunc_GetMethodSignature(PyObject* func);

#endif /* PyObjC_FUNCTION_H */
