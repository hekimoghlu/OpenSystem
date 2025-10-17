/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 7, 2022.
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
#ifndef NI_MEASURE_H
#define NI_MEASURE_H

#include "nd_image.h"

/* structure for array regions to find objects: */
typedef struct {
    int start[NI_MAXDIM], end[NI_MAXDIM];
} NI_ObjectRegion;

int NI_FindObjects(PyArrayObject*, npy_intp, npy_intp*);

int NI_CenterOfMass(PyArrayObject*, PyArrayObject*, npy_intp, npy_intp,
                    npy_intp*, npy_intp, double*);

int NI_Histogram(PyArrayObject*, PyArrayObject*, npy_intp, npy_intp,
                 npy_intp*, npy_intp, PyArrayObject**, double, double,
                 npy_intp);

int NI_Statistics(PyArrayObject*, PyArrayObject*, npy_intp, npy_intp,
                  npy_intp*, npy_intp, double*, npy_intp*, double*,
                  double*, double*, npy_intp*, npy_intp*);

int NI_WatershedIFT(PyArrayObject*, PyArrayObject*, PyArrayObject*, 
                                        PyArrayObject*);

#endif
