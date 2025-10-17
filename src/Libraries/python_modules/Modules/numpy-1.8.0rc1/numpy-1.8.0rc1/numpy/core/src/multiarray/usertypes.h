/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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

#ifndef _NPY_PRIVATE_USERTYPES_H_
#define _NPY_PRIVATE_USERTYPES_H_

#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT PyArray_Descr **userdescrs;
#else
NPY_NO_EXPORT PyArray_Descr **userdescrs;
#endif

NPY_NO_EXPORT void
PyArray_InitArrFuncs(PyArray_ArrFuncs *f);

NPY_NO_EXPORT int
PyArray_RegisterCanCast(PyArray_Descr *descr, int totype,
                        NPY_SCALARKIND scalar);

NPY_NO_EXPORT int
PyArray_RegisterDataType(PyArray_Descr *descr);

NPY_NO_EXPORT int
PyArray_RegisterCastFunc(PyArray_Descr *descr, int totype,
                         PyArray_VectorUnaryFunc *castfunc);

#endif
