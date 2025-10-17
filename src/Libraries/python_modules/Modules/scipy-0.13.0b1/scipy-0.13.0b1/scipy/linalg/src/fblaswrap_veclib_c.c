/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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

#include <Accelerate/Accelerate.h>

//#define WRAP_F77(a) wcblas_##a##_
#define WRAP_F77(a) w##a##_
void WRAP_F77(cdotc)(void *dotc, const int *N, const void *X, const int *incX, 
		     const void *Y, const int *incY)
{
    cblas_cdotc_sub(*N, X, *incX, Y, *incY, dotc);
}

void WRAP_F77(cdotu)(void *dotu, const int *N, const void *X, const int *incX,
		     const void *Y, const int *incY)
{
    cblas_cdotu_sub(*N, X, *incX, Y, *incY, dotu);
}

void WRAP_F77(zdotc)(void *dotu, const int *N, const void *X, const int *incX,
		     const void *Y, const int *incY)
{
    cblas_zdotc_sub(*N, X, *incX, Y, *incY, dotu);
}
void WRAP_F77(zdotu)(void *dotu, const int *N, const void *X, const int *incX,
		     const void *Y, const int *incY)
{
    cblas_zdotu_sub(*N, X, *incX, Y, *incY, dotu);
}

float WRAP_F77(sdot)(const int *N, const float *X, const int *incX,
		     const float *Y, const int *incY)
{
    return cblas_sdot(*N, X, *incX, Y, *incY);
}

float WRAP_F77(sasum)(const int *N, const float *X, const int *incX)
{
    return cblas_sasum(*N, X, *incX);
}

float WRAP_F77(scasum)(const int *N, const void *X, const int *incX)
{
    return cblas_scasum(*N, X, *incX);
}

float WRAP_F77(snrm2)(const int *N, const float *X, const int *incX)
{
    return cblas_snrm2(*N, X, *incX);
}

float WRAP_F77(scnrm2)(const int *N, const void *X, const int *incX)
{
    return cblas_scnrm2(*N, X, *incX);
}

