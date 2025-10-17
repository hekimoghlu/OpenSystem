/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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
/* { dg-options -mlong-double-128 { target powerpc64*-*-* } } */

#include "ffitest.h"

static int floating(int a, float b, double c, long double d, int e)
{
  int i;

  printf("%d\n", e);
  printf("a:%d b:%f c:%f d:%llf e:%d\n", a, b, c, d, e);
  i = (int) ((float)a/b + ((float)c/(float)d));

  return i;
}

int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  ffi_arg rint;

  float f;
  signed int si1;
  double d;
  long double ld;
  signed int si2;

  args[0] = &ffi_type_sint;
  values[0] = &si1;
  args[1] = &ffi_type_float;
  values[1] = &f;
  args[2] = &ffi_type_double;
  values[2] = &d;
  args[3] = &ffi_type_longdouble;
  values[3] = &ld;
  args[4] = &ffi_type_sint;
  values[4] = &si2;

  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 5,
		     &ffi_type_sint, args) == FFI_OK);

  si1 = 6;
  f = 3.14159;
  d = (double)1.0/(double)3.0;
  ld = 2.71828182846L;
  si2 = 10;

  floating (si1, f, d, ld, si2);

  printf("calling through ffi %d\n", si2);
  ffi_call(&cif, FFI_FN(floating), &rint, values);

  printf ("%d vs %d\n", (int)rint, floating (si1, f, d, ld, si2));

  CHECK(rint == floating(si1, f, d, ld, si2));

  exit (0);
}
