/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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
#include "ffitest.h"
#include <complex.h>

static _Complex T_C_TYPE return_c(_Complex T_C_TYPE c1, float fl2, unsigned int in3, _Complex T_C_TYPE c4)
{
  return c1 + fl2 + in3 + c4;
}
int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  _Complex T_C_TYPE c1, c4, rc, rc2;
  float fl2;
  unsigned int in3;
  args[0] = &T_FFI_TYPE;
  args[1] = &ffi_type_float;
  args[2] = &ffi_type_uint;
  args[3] = &T_FFI_TYPE;
  values[0] = &c1;
  values[1] = &fl2;
  values[2] = &in3;
  values[3] = &c4;

  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 4,
		     &T_FFI_TYPE, args) == FFI_OK);
  c1 = 127.0 + 255.0 * I;
  fl2 = 128.0;
  in3 = 255;
  c4 = 512.7 + 1024.1 * I;

  ffi_call(&cif, FFI_FN(return_c), &rc, values);
  rc2 = return_c(c1, fl2, in3, c4);
  printf ("%f,%fi vs %f,%fi\n",
	  T_CONV creal (rc), T_CONV cimag (rc),
	  T_CONV creal (rc2), T_CONV cimag (rc2));
  CHECK(rc == rc2);
  exit(0);
}
