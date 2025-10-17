/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 17, 2024.
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
/* { dg-options "-Wno-format" { target alpha*-dec-osf* } } */
#include "ffitest.h"
static long long return_ll(int ll0, long long ll1, int ll2)
{
  return ll0 + ll1 + ll2;
}

int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  long long rlonglong;
  long long ll1;
  unsigned ll0, ll2;

  args[0] = &ffi_type_sint;
  args[1] = &ffi_type_sint64;
  args[2] = &ffi_type_sint;
  values[0] = &ll0;
  values[1] = &ll1;
  values[2] = &ll2;

  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 3,
		     &ffi_type_sint64, args) == FFI_OK);

  ll0 = 11111111;
  ll1 = 11111111111000LL;
  ll2 = 11111111;

  ffi_call(&cif, FFI_FN(return_ll), &rlonglong, values);
  printf("res: %" PRIdLL ", %" PRIdLL "\n", rlonglong, ll0 + ll1 + ll2);
  /* { dg-output "res: 11111133333222, 11111133333222" } */
  exit(0);
}
