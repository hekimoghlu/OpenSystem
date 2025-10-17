/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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
#include "float.h"

typedef union
{
  double d;
  unsigned char c[sizeof (double)];
} value_type;

#define CANARY 0xba

static double dblit(float f)
{
  return f/3.0;
}

int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  float f;
  value_type result[2];
  unsigned int i;

  args[0] = &ffi_type_float;
  values[0] = &f;

  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 1,
		     &ffi_type_double, args) == FFI_OK);

  f = 3.14159;

  /* Put a canary in the return array.  This is a regression test for
     a buffer overrun.  */
  memset(result[1].c, CANARY, sizeof (double));

  ffi_call(&cif, FFI_FN(dblit), &result[0].d, values);

  /* These are not always the same!! Check for a reasonable delta */

  CHECK(result[0].d - dblit(f) < DBL_EPSILON);

  /* Check the canary.  */
  for (i = 0; i < sizeof (double); ++i)
    CHECK(result[1].c[i] == CANARY);

  exit(0);

}
