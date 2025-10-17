/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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

#ifndef _NPY_PRIVATE__ITEM_SELECTION_H_
#define _NPY_PRIVATE__ITEM_SELECTION_H_

/*
 * Counts the number of True values in a raw boolean array. This
 * is a low-overhead function which does no heap allocations.
 *
 * Returns -1 on error.
 */
NPY_NO_EXPORT npy_intp
count_boolean_trues(int ndim, char *data, npy_intp *ashape, npy_intp *astrides);

/*
 * Gets a single item from the array, based on a single multi-index
 * array of values, which must be of length PyArray_NDIM(self).
 */
NPY_NO_EXPORT PyObject *
PyArray_MultiIndexGetItem(PyArrayObject *self, npy_intp *multi_index);

/*
 * Sets a single item in the array, based on a single multi-index
 * array of values, which must be of length PyArray_NDIM(self).
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
PyArray_MultiIndexSetItem(PyArrayObject *self, npy_intp *multi_index,
                                                PyObject *obj);

#endif
