/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 5, 2022.
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

static size_t ABI_ATTR my_strlen(char *s)
{
  return (strlen(s));
}

int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  ffi_arg rint;
  char *s;

  args[0] = &ffi_type_pointer;
  values[0] = (void*) &s;
  
  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, ABI_NUM, 1,
		     &ffi_type_sint, args) == FFI_OK);
  
  s = "a";
  ffi_call(&cif, FFI_FN(my_strlen), &rint, values);
  CHECK(rint == 1);
  
  s = "1234567";
  ffi_call(&cif, FFI_FN(my_strlen), &rint, values);
  CHECK(rint == 7);
  
  s = "1234567890123456789012345";
  ffi_call(&cif, FFI_FN(my_strlen), &rint, values);
  CHECK(rint == 25);
  
  exit (0);
}
  

