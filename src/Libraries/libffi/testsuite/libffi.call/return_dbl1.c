/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 3, 2025.
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

static double return_dbl(double dbl1, float fl2, unsigned int in3, double dbl4)
{
  return dbl1 + fl2 + in3 + dbl4;
}
int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  double dbl1, dbl4, rdbl;
  float fl2;
  unsigned int in3;
  args[0] = &ffi_type_double;
  args[1] = &ffi_type_float;
  args[2] = &ffi_type_uint;
  args[3] = &ffi_type_double;
  values[0] = &dbl1;
  values[1] = &fl2;
  values[2] = &in3;
  values[3] = &dbl4;

  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 4,
		     &ffi_type_double, args) == FFI_OK);
  dbl1 = 127.0;
  fl2 = 128.0;
  in3 = 255;
  dbl4 = 512.7;

  ffi_call(&cif, FFI_FN(return_dbl), &rdbl, values);
  printf ("%f vs %f\n", rdbl, return_dbl(dbl1, fl2, in3, dbl4));
  CHECK(rdbl ==  dbl1 + fl2 + in3 + dbl4);
  exit(0);
}

