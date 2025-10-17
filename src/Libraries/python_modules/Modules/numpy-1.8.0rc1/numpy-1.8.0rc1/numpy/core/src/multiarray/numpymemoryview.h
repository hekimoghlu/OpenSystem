/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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

#ifndef _NPY_PRIVATE_NUMPYMEMORYVIEW_H_
#define _NPY_PRIVATE_NUMPYMEMORYVIEW_H_

/*
 * Memoryview is introduced to 2.x series only in 2.7, so for supporting 2.6,
 * we need to have a minimal implementation here.
 */
#if PY_VERSION_HEX < 0x02070000

typedef struct {
    PyObject_HEAD
    PyObject *base;
    Py_buffer view;
} PyMemorySimpleViewObject;

NPY_NO_EXPORT PyObject *
PyMemorySimpleView_FromObject(PyObject *base);

#define PyMemorySimpleView_GET_BUFFER(op) (&((PyMemorySimpleViewObject *)(op))->view)

#define PyMemoryView_FromObject PyMemorySimpleView_FromObject
#define PyMemoryView_GET_BUFFER PyMemorySimpleView_GET_BUFFER

#endif

NPY_NO_EXPORT int
_numpymemoryview_init(PyObject **typeobject);

#endif
