/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 29, 2023.
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

static float ABI_ATTR align_arguments(int i1,
                                      double f2,
                                      int i3,
                                      double f4)
{
  return i1+f2+i3+f4;
}

int main(void)
{
  ffi_cif cif;
  ffi_type *args[4] = {
    &ffi_type_sint,
    &ffi_type_double,
    &ffi_type_sint,
    &ffi_type_double
  };
  double fa[2] = {1,2};
  int ia[2] = {1,2};
  void *values[4] = {&ia[0], &fa[0], &ia[1], &fa[1]};
  float f, ff;

  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, ABI_NUM, 4,
		     &ffi_type_float, args) == FFI_OK);

  ff = align_arguments(ia[0], fa[0], ia[1], fa[1]);;

  ffi_call(&cif, FFI_FN(align_arguments), &f, values);

  if (f == ff)
    printf("align arguments tests ok!\n");
  else
    CHECK(0);
  exit(0);
}

