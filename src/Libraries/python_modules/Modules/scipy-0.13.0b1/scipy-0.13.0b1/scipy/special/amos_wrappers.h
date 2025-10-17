/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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
#ifndef _AMOS_WRAPPERS_H
#define _AMOS_WRAPPERS_H
#include "Python.h"
#include "sf_error.h"

#include <numpy/npy_math.h>

#define DO_SFERR(name, varp)                          \
    do {                                              \
      if (nz !=0 || ierr != 0) {                      \
        sf_error(name, ierr_to_sferr(nz, ierr), NULL);\
        set_nan_if_no_computation_done(varp, ierr);   \
      }                                               \
    } while (0)

int ierr_to_sferr( int nz, int ierr);
void set_nan_if_no_computation_done(npy_cdouble *var, int ierr);
int airy_wrap(double x, double *ai, double *aip, double *bi, double *bip);
int cairy_wrap(npy_cdouble z, npy_cdouble *ai, npy_cdouble *aip, npy_cdouble *bi, npy_cdouble *bip);
int cairy_wrap_e(npy_cdouble z, npy_cdouble *ai, npy_cdouble *aip, npy_cdouble *bi, npy_cdouble *bip);
int cairy_wrap_e_real(double z, double *ai, double *aip, double *bi, double *bip);
npy_cdouble cbesi_wrap( double v, npy_cdouble z);
npy_cdouble cbesi_wrap_e( double v, npy_cdouble z);
double cbesi_wrap_e_real( double v, double z);
npy_cdouble cbesj_wrap( double v, npy_cdouble z);
npy_cdouble cbesj_wrap_e( double v, npy_cdouble z);
double cbesj_wrap_real(double v, double z);
double cbesj_wrap_e_real( double v, double z);
npy_cdouble cbesy_wrap( double v, npy_cdouble z);
double cbesy_wrap_real(double v, double x);
npy_cdouble cbesy_wrap_e( double v, npy_cdouble z);
double cbesy_wrap_e_real( double v, double z);
npy_cdouble cbesk_wrap( double v, npy_cdouble z);
npy_cdouble cbesk_wrap_e( double v, npy_cdouble z);  
double cbesk_wrap_real( double v, double z);
double cbesk_wrap_e_real( double v, double z);
double cbesk_wrap_real_int(int n, double z);
npy_cdouble cbesh_wrap1( double v, npy_cdouble z);
npy_cdouble cbesh_wrap1_e( double v, npy_cdouble z);  
npy_cdouble cbesh_wrap2( double v, npy_cdouble z);
npy_cdouble cbesh_wrap2_e( double v, npy_cdouble z);
/* 
int cairy_(double *, int *, int *, doublecomplex *, int *, int *);
int cbiry_(doublecomplex *, int *, int *, doublecomplex *, int *, int *);
int cbesi_(doublecomplex *, double *, int *, int *, doublecomplex *, int *, int *);
int cbesj_(doublecomplex *, double *, int *, int *, doublecomplex *, int *, int *);
int cbesk_(doublecomplex *, double *, int *, int *, doublecomplex *, int *, int *);
int cbesy_(doublecomplex *, double *, int *, int *, doublecomplex *, int *, doublecomplex *, int *);
int cbesh_(doublecomplex *, double *, int *, int *, int *, doublecomplex *, int *, int *);
*/

#endif



  







