/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 29, 2025.
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
static unsigned long return_ul(unsigned long ul1, unsigned long ul2)
{
  return ul1 + ul2;
}

int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  ffi_arg res;
  unsigned long ul1, ul2;

  args[0] = &ffi_type_ulong;
  args[1] = &ffi_type_ulong;
  values[0] = &ul1;
  values[1] = &ul2;

  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 2,
		     &ffi_type_ulong, args) == FFI_OK);

  ul1 = 1073741823L;
  ul2 = 1073741824L;

  ffi_call(&cif, FFI_FN(return_ul), &res, values);
  printf("res: %lu, %lu\n", (unsigned long)res, ul1 + ul2);
  /* { dg-output "res: 2147483647, 2147483647" } */

  exit(0);
}
