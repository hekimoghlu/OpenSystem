/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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

#ifndef _NPY_CALCULATION_H_
#define _NPY_CALCULATION_H_

NPY_NO_EXPORT PyObject*
PyArray_ArgMax(PyArrayObject* self, int axis, PyArrayObject *out);

NPY_NO_EXPORT PyObject*
PyArray_ArgMin(PyArrayObject* self, int axis, PyArrayObject *out);

NPY_NO_EXPORT PyObject*
PyArray_Max(PyArrayObject* self, int axis, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_Min(PyArrayObject* self, int axis, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_Ptp(PyArrayObject* self, int axis, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_Mean(PyArrayObject* self, int axis, int rtype, PyArrayObject* out);

NPY_NO_EXPORT PyObject *
PyArray_Round(PyArrayObject *a, int decimals, PyArrayObject *out);

NPY_NO_EXPORT PyObject*
PyArray_Trace(PyArrayObject* self, int offset, int axis1, int axis2,
                int rtype, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_Clip(PyArrayObject* self, PyObject* min, PyObject* max, PyArrayObject *out);

NPY_NO_EXPORT PyObject*
PyArray_Conjugate(PyArrayObject* self, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_Round(PyArrayObject* self, int decimals, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_Std(PyArrayObject* self, int axis, int rtype, PyArrayObject* out,
                int variance);

NPY_NO_EXPORT PyObject *
__New_PyArray_Std(PyArrayObject *self, int axis, int rtype, PyArrayObject *out,
                  int variance, int num);

NPY_NO_EXPORT PyObject*
PyArray_Sum(PyArrayObject* self, int axis, int rtype, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_CumSum(PyArrayObject* self, int axis, int rtype, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_Prod(PyArrayObject* self, int axis, int rtype, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_CumProd(PyArrayObject* self, int axis, int rtype, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_All(PyArrayObject* self, int axis, PyArrayObject* out);

NPY_NO_EXPORT PyObject*
PyArray_Any(PyArrayObject* self, int axis, PyArrayObject* out);

#endif
