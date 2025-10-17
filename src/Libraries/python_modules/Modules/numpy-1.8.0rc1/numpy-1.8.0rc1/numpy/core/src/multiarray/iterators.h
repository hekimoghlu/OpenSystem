/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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

#ifndef _NPY_ARRAYITERATORS_H_
#define _NPY_ARRAYITERATORS_H_

/*
 * Parses an index that has no fancy indexing. Populates
 * out_dimensions, out_strides, and out_offset.
 */
NPY_NO_EXPORT int
parse_index(PyArrayObject *self, PyObject *op,
            npy_intp *out_dimensions,
            npy_intp *out_strides,
            npy_intp *out_offset,
            int check_index);

NPY_NO_EXPORT PyObject
*iter_subscript(PyArrayIterObject *, PyObject *);

NPY_NO_EXPORT int
iter_ass_subscript(PyArrayIterObject *, PyObject *, PyObject *);

NPY_NO_EXPORT int
slice_GetIndices(PySliceObject *r, npy_intp length,
                 npy_intp *start, npy_intp *stop, npy_intp *step,
                 npy_intp *slicelength);

#endif
