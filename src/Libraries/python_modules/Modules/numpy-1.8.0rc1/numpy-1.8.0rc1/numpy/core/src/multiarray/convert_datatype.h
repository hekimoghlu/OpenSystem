/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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

#ifndef _NPY_ARRAY_CONVERT_DATATYPE_H_
#define _NPY_ARRAY_CONVERT_DATATYPE_H_

NPY_NO_EXPORT PyArray_VectorUnaryFunc *
PyArray_GetCastFunc(PyArray_Descr *descr, int type_num);

NPY_NO_EXPORT int
PyArray_ObjectType(PyObject *op, int minimum_type);

NPY_NO_EXPORT PyArrayObject **
PyArray_ConvertToCommonType(PyObject *op, int *retn);

NPY_NO_EXPORT int
PyArray_ValidType(int type);

/* Like PyArray_CanCastArrayTo */
NPY_NO_EXPORT npy_bool
can_cast_scalar_to(PyArray_Descr *scal_type, char *scal_data,
                    PyArray_Descr *to, NPY_CASTING casting);

/*
 * This function calls Py_DECREF on flex_dtype, and replaces it with
 * a new dtype that has been adapted based on the values in data_dtype
 * and data_obj. If the flex_dtype is not flexible, it leaves it as is.
 *
 * The current flexible dtypes include NPY_STRING, NPY_UNICODE, NPY_VOID,
 * and NPY_DATETIME with generic units.
 */
NPY_NO_EXPORT void
PyArray_AdaptFlexibleDType(PyObject *data_obj, PyArray_Descr *data_dtype,
                            PyArray_Descr **flex_dtype);

#endif
