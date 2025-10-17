/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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
#ifndef NI_FILTERS_H
#define NI_FILTERS_H

int NI_Correlate1D(PyArrayObject*, PyArrayObject*, int, PyArrayObject*,
                   NI_ExtendMode, double, npy_intp);
int NI_Correlate(PyArrayObject*, PyArrayObject*, PyArrayObject*,
                 NI_ExtendMode, double, npy_intp*);
int NI_UniformFilter1D(PyArrayObject*, npy_intp, int, PyArrayObject*,
                       NI_ExtendMode, double, npy_intp);
int NI_MinOrMaxFilter1D(PyArrayObject*, npy_intp, int, PyArrayObject*,
                        NI_ExtendMode, double, npy_intp, int);
int NI_MinOrMaxFilter(PyArrayObject*, PyArrayObject*, PyArrayObject*,
                      PyArrayObject*, NI_ExtendMode, double, npy_intp*,
                                            int);
int NI_RankFilter(PyArrayObject*, int, PyArrayObject*, PyArrayObject*,
                                    NI_ExtendMode, double, npy_intp*);
int NI_GenericFilter1D(PyArrayObject*, int (*)(double*, npy_intp,
                       double*, npy_intp, void*), void*, npy_intp, int,
                       PyArrayObject*, NI_ExtendMode, double, npy_intp);
int NI_GenericFilter(PyArrayObject*, int (*)(double*, npy_intp, double*,
                                         void*), void*, PyArrayObject*, PyArrayObject*,
                     NI_ExtendMode, double, npy_intp*);
#endif
