/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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

static size_t ABI_ATTR my_f(float a, char *s, int i)
{
  return (size_t) ((int) strlen(s) + (int) a + i);
}

int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  ffi_arg rint;
  char *s;
  int v1;
  float v2;
  args[2] = &ffi_type_sint;
  args[1] = &ffi_type_pointer;
  args[0] = &ffi_type_float;
  values[2] = (void*) &v1;
  values[1] = (void*) &s;
  values[0] = (void*) &v2;
  
  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, ABI_NUM, 3,
		       &ffi_type_sint, args) == FFI_OK);
  
  s = "a";
  v1 = 1;
  v2 = 0.0;
  ffi_call(&cif, FFI_FN(my_f), &rint, values);
  CHECK(rint == 2);
  
  s = "1234567";
  v2 = -1.0;
  v1 = -2;
  ffi_call(&cif, FFI_FN(my_f), &rint, values);
  CHECK(rint == 4);
  
  s = "1234567890123456789012345";
  v2 = 1.0;
  v1 = 2;
  ffi_call(&cif, FFI_FN(my_f), &rint, values);
  CHECK(rint == 28);
  
  exit(0);
}

