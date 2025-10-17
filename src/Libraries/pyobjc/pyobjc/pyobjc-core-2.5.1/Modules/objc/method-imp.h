/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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
#ifndef PyObjC_METHOD_IMP_H
#define PyObjC_METHOD_IMP_H

extern PyTypeObject PyObjCIMP_Type;

#define PyObjCIMP_Check(obj) PyObject_TypeCheck(obj, &PyObjCIMP_Type)

extern PyObject* PyObjCIMP_New(
		IMP imp, 
		SEL sel,
		PyObjC_CallFunc callfunc,
		PyObjCMethodSignature* signature,
		int flags);
extern IMP PyObjCIMP_GetIMP(PyObject* self);
extern PyObjC_CallFunc PyObjCIMP_GetCallFunc(PyObject* self);
extern PyObjCMethodSignature* PyObjCIMP_GetSignature(PyObject* self);
extern int PyObjCIMP_GetFlags(PyObject* self);
extern SEL PyObjCIMP_GetSelector(PyObject* self);

extern int PyObjCIMP_SetUpMethodWrappers(void);

#endif /* PyObjC_METHOD_IMP_H */
