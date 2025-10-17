/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 21, 2025.
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

#ifndef _NPY_ARRAY_NUMBER_H_
#define _NPY_ARRAY_NUMBER_H_

typedef struct {
    PyObject *add;
    PyObject *subtract;
    PyObject *multiply;
    PyObject *divide;
    PyObject *remainder;
    PyObject *power;
    PyObject *square;
    PyObject *reciprocal;
    PyObject *_ones_like;
    PyObject *sqrt;
    PyObject *negative;
    PyObject *absolute;
    PyObject *invert;
    PyObject *left_shift;
    PyObject *right_shift;
    PyObject *bitwise_and;
    PyObject *bitwise_xor;
    PyObject *bitwise_or;
    PyObject *less;
    PyObject *less_equal;
    PyObject *equal;
    PyObject *not_equal;
    PyObject *greater;
    PyObject *greater_equal;
    PyObject *floor_divide;
    PyObject *true_divide;
    PyObject *logical_or;
    PyObject *logical_and;
    PyObject *floor;
    PyObject *ceil;
    PyObject *maximum;
    PyObject *minimum;
    PyObject *rint;
    PyObject *conjugate;
} NumericOps;

#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT NumericOps n_ops;
extern NPY_NO_EXPORT PyNumberMethods array_as_number;
#else
NPY_NO_EXPORT NumericOps n_ops;
NPY_NO_EXPORT PyNumberMethods array_as_number;
#endif

NPY_NO_EXPORT PyObject *
array_int(PyArrayObject *v);

NPY_NO_EXPORT int
PyArray_SetNumericOps(PyObject *dict);

NPY_NO_EXPORT PyObject *
PyArray_GetNumericOps(void);

NPY_NO_EXPORT PyObject *
PyArray_GenericBinaryFunction(PyArrayObject *m1, PyObject *m2, PyObject *op);

NPY_NO_EXPORT PyObject *
PyArray_GenericUnaryFunction(PyArrayObject *m1, PyObject *op);

NPY_NO_EXPORT PyObject *
PyArray_GenericReduceFunction(PyArrayObject *m1, PyObject *op, int axis,
                              int rtype, PyArrayObject *out);

NPY_NO_EXPORT PyObject *
PyArray_GenericAccumulateFunction(PyArrayObject *m1, PyObject *op, int axis,
                                  int rtype, PyArrayObject *out);

#endif
