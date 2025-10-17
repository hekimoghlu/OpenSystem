/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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

#ifndef _NPY_PRIVATE__UFUNC_TYPE_RESOLUTION_H_
#define _NPY_PRIVATE__UFUNC_TYPE_RESOLUTION_H_

NPY_NO_EXPORT int
PyUFunc_SimpleBinaryComparisonTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_SimpleUnaryOperationTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_OnesLikeTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_SimpleBinaryOperationTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_AbsoluteTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_AdditionTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_SubtractionTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_MultiplicationTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes);

NPY_NO_EXPORT int
PyUFunc_DivisionTypeResolver(PyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes);

/*
 * Does a linear search for the best inner loop of the ufunc.
 *
 * Note that if an error is returned, the caller must free the non-zero
 * references in out_dtype.  This function does not do its own clean-up.
 */
NPY_NO_EXPORT int
linear_search_type_resolver(PyUFuncObject *self,
                        PyArrayObject **op,
                        NPY_CASTING input_casting,
                        NPY_CASTING output_casting,
                        int any_object,
                        PyArray_Descr **out_dtype);

/*
 * Does a linear search for the inner loop of the ufunc specified by type_tup.
 *
 * Note that if an error is returned, the caller must free the non-zero
 * references in out_dtype.  This function does not do its own clean-up.
 */
NPY_NO_EXPORT int
type_tuple_type_resolver(PyUFuncObject *self,
                        PyObject *type_tup,
                        PyArrayObject **op,
                        NPY_CASTING casting,
                        int any_object,
                        PyArray_Descr **out_dtype);

NPY_NO_EXPORT int
PyUFunc_DefaultLegacyInnerLoopSelector(PyUFuncObject *ufunc,
                                PyArray_Descr **dtypes,
                                PyUFuncGenericFunction *out_innerloop,
                                void **out_innerloopdata,
                                int *out_needs_api);

NPY_NO_EXPORT int
PyUFunc_DefaultMaskedInnerLoopSelector(PyUFuncObject *ufunc,
                            PyArray_Descr **dtypes,
                            PyArray_Descr *mask_dtypes,
                            npy_intp *NPY_UNUSED(fixed_strides),
                            npy_intp NPY_UNUSED(fixed_mask_stride),
                            PyUFunc_MaskedStridedInnerLoopFunc **out_innerloop,
                            NpyAuxData **out_innerloopdata,
                            int *out_needs_api);


#endif
