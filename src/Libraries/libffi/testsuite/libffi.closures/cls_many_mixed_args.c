/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 25, 2025.
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
/* { dg-do run } */
#include "ffitest.h"
#include <float.h>
#include <math.h>

#define NARGS 16

static void cls_ret_double_fn(ffi_cif* cif __UNUSED__, void* resp, void** args,
			      void* userdata __UNUSED__)
{
  int i;
  double r = 0;
  double t;
  for(i = 0; i < NARGS; i++)
    {
    if(i == 4 || i == 9 || i == 11 || i == 13 || i == 15)
      {
      t = *(long int *)args[i];
      CHECK(t == i+1);
      }
    else
      {
      t = *(double *)args[i];
      CHECK(fabs(t - ((i+1) * 0.1)) < FLT_EPSILON);
      }
    r += t;
    }
  *(double *)resp = r;
}
typedef double (*cls_ret_double)(double, double, double, double, long int,
double, double, double, double, long int, double, long int, double, long int,
double, long int);

int main (void)
{
  ffi_cif cif;
  void *code;
  ffi_closure *pcl = ffi_closure_alloc(sizeof(ffi_closure), &code);
  ffi_type * cl_arg_types[NARGS];
  double res;
  int i;
  double expected = 64.9;

  for(i = 0; i < NARGS; i++)
    {
    if(i == 4 || i == 9 || i == 11 || i == 13 || i == 15)
      cl_arg_types[i] = &ffi_type_slong;
    else
      cl_arg_types[i] = &ffi_type_double;
    }

  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, NARGS,
		     &ffi_type_double, cl_arg_types) == FFI_OK);

  CHECK(ffi_prep_closure_loc(pcl, &cif, cls_ret_double_fn, NULL, code) == FFI_OK);

  res = (((cls_ret_double)code))(0.1, 0.2, 0.3, 0.4, 5, 0.6, 0.7, 0.8, 0.9, 10,
                                 1.1, 12, 1.3, 14, 1.5, 16);
  if (fabs(res - expected) < FLT_EPSILON)
    exit(0);
  else
    abort();
}

