/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
#ifndef NI_MORPHOLOGY_H
#define NI_MORPHOLOGY_H

int NI_BinaryErosion(PyArrayObject*, PyArrayObject*, PyArrayObject*, 
         PyArrayObject*, int, npy_intp*, int, int, int*, NI_CoordinateList**);
int NI_BinaryErosion2(PyArrayObject*, PyArrayObject*, PyArrayObject*,
                      int, npy_intp*, int, NI_CoordinateList**);
int NI_DistanceTransformBruteForce(PyArrayObject*, int, PyArrayObject*,
                                                                     PyArrayObject*, PyArrayObject*);
int NI_DistanceTransformOnePass(PyArrayObject*, PyArrayObject *,
                                                                PyArrayObject*);
int NI_EuclideanFeatureTransform(PyArrayObject*, PyArrayObject*, 
                                                                 PyArrayObject*);

#endif
