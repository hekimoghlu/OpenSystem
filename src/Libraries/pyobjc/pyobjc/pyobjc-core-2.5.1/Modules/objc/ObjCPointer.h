/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 16, 2025.
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

#ifndef PyObjC_OBJC_POINTER_H
#define PyObjC_OBJC_POINTER_H

/* Python wrapper around C pointer 
 *
 * NOTE: This class is almost never used, pointers in method interfaces are,
 * or should be, treated differently and I've yet to run into a Cocoa structure 
 * that contains pointers.
 */

typedef struct
{
  PyObject_VAR_HEAD

  void *ptr;
  PyObject *type;
  char contents[1];
} PyObjCPointer;

extern int	PyObjCPointer_RaiseException;

extern PyTypeObject PyObjCPointer_Type;

#define PyObjCPointer_Check(o) (Py_TYPE(o) == &PyObjCPointer_Type)

extern PyObjCPointer *PyObjCPointer_New(void *ptr, const char *type);
#define PyObjCPointer_Ptr(obj) (((PyObjCPointer*)(obj))->ptr)

#endif /* PyObjC_OBJC_POINTER_H */
