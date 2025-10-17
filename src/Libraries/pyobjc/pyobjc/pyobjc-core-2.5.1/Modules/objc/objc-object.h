/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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

#ifndef PyObjC_OBJC_OBJECT_H
#define PyObjC_OBJC_OBJECT_H

#define PyObjCObject_kDEFAULT 0x00
#define PyObjCObject_kUNINITIALIZED 	0x01
#define PyObjCObject_kCLASSIC 		0x02
#define PyObjCObject_kDEALLOC_HELPER	0x04
#define PyObjCObject_kSHOULD_NOT_RELEASE      0x08
#define PyObjCObject_kMAGIC_COOKIE      0x10
#define PyObjCObject_kCFOBJECT      0x20
#define PyObjCObject_kBLOCK      0x40

typedef struct {
	PyObject_HEAD
	__strong id objc_object;
	int 	    flags;
} PyObjCObject;

typedef struct {
	PyObject_HEAD
	__strong id objc_object;
	int 	    flags;
	PyObjCMethodSignature* signature;
} PyObjCBlockObject;


extern PyObjCClassObject PyObjCObject_Type;
#define PyObjCObject_Check(obj) PyObject_TypeCheck(obj, (PyTypeObject*)&PyObjCObject_Type)

PyObject* PyObjCObject_New(id objc_object, int flags, int retain);
PyObject* PyObjCObject_FindSelector(PyObject* cls, SEL selector);
id 	  PyObjCObject_GetObject(PyObject* object);
void 	  PyObjCObject_ClearObject(PyObject* object);
#define   PyObjCObject_GetObject(object) (((PyObjCObject*)(object))->objc_object)
void _PyObjCObject_FreeDeallocHelper(PyObject* obj);
PyObject* _PyObjCObject_NewDeallocHelper(id objc_object);
#define PyObjCObject_GetFlags(object) (((PyObjCObject*)(object))->flags)
#define PyObjCObject_IsClassic(object) (PyObjCObject_GetFlags(object) & PyObjCObject_kCLASSIC)
#define PyObjCObject_IsBlock(object) (PyObjCObject_GetFlags(object) & PyObjCObject_kBLOCK)
#define PyObjCObject_GetBlock(object) (((PyObjCBlockObject*)(object))->signature)
#define PyObjCObject_SET_BLOCK(object, value) (((PyObjCBlockObject*)(object))->signature = (value))

PyObject* PyObjCObject_GetAttr(PyObject* object, PyObject* key);
PyObject* PyObjCObject_GetAttrString(PyObject* object, char* key);


PyObject* PyObjCObject_NewTransient(id objc_object, int* cookie);
void PyObjCObject_ReleaseTransient(PyObject* proxy, int cookie);

#endif /* PyObjC_OBJC_OBJECT_H */
