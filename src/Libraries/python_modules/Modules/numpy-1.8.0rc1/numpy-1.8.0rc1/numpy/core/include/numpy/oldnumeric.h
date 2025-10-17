/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 30, 2025.
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

#include "arrayobject.h"

#ifndef REFCOUNT
#  define REFCOUNT NPY_REFCOUNT
#  define MAX_ELSIZE 16
#endif

#define PyArray_UNSIGNED_TYPES
#define PyArray_SBYTE NPY_BYTE
#define PyArray_CopyArray PyArray_CopyInto
#define _PyArray_multiply_list PyArray_MultiplyIntList
#define PyArray_ISSPACESAVER(m) NPY_FALSE
#define PyScalarArray_Check PyArray_CheckScalar

#define CONTIGUOUS NPY_CONTIGUOUS
#define OWN_DIMENSIONS 0
#define OWN_STRIDES 0
#define OWN_DATA NPY_OWNDATA
#define SAVESPACE 0
#define SAVESPACEBIT 0

#undef import_array
#define import_array() { if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); } }
