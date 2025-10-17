/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 26, 2022.
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

static int checking(int a, short b, signed char c)
{

  return (a < 0 && b < 0 && c < 0);
}

int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  ffi_arg rint;

  signed int si;
  signed short ss;
  signed char sc;

  args[0] = &ffi_type_sint;
  values[0] = &si;
  args[1] = &ffi_type_sshort;
  values[1] = &ss;
  args[2] = &ffi_type_schar;
  values[2] = &sc;

  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 3,
		     &ffi_type_sint, args) == FFI_OK);

  si = -6;
  ss = -12;
  sc = -1;

  checking (si, ss, sc);

  ffi_call(&cif, FFI_FN(checking), &rint, values);

  printf ("%d vs %d\n", (int)rint, checking (si, ss, sc));

  CHECK(rint != 0);

  exit (0);
}

