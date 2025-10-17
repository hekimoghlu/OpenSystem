/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 15, 2025.
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

static _Complex T_C_TYPE return_c(_Complex T_C_TYPE c)
{
  printf ("%f,%fi\n", T_CONV creal (c), T_CONV cimag (c));
  return 2 * c;
}
int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  _Complex T_C_TYPE c, rc, rc2;
  T_C_TYPE cr, ci;

  args[0] = &T_FFI_TYPE;
  values[0] = &c;

  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 1,
		     &T_FFI_TYPE, args) == FFI_OK);

  for (cr = -127.0; cr <  127; cr++)
    {
      ci = 1000.0 - cr;
      c = cr + ci * I;
      ffi_call(&cif, FFI_FN(return_c), &rc, values);
      rc2 = return_c(c);
      printf ("%f,%fi vs %f,%fi\n",
	      T_CONV creal (rc), T_CONV cimag (rc),
	      T_CONV creal (rc2), T_CONV cimag (rc2));
      CHECK(rc == 2 * c);
    }
  exit(0);
}
