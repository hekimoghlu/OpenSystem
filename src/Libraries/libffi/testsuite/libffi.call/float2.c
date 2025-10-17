/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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

#include <math.h>

static long double ldblit(float f)
{
  return (long double) (((long double) f)/ (long double) 3.0);
}

int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  float f;
  long double ld;
  long double original;

  args[0] = &ffi_type_float;
  values[0] = &f;

  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 1,
		     &ffi_type_longdouble, args) == FFI_OK);

  f = 3.14159;

#if defined(__sun) && defined(__GNUC__)
  /* long double support under SunOS/gcc is pretty much non-existent.
     You'll get the odd bus error in library routines like printf() */
#else
  printf ("%Lf\n", ldblit(f));
#endif

  ld = 666;
  ffi_call(&cif, FFI_FN(ldblit), &ld, values);

#if defined(__sun) && defined(__GNUC__)
  /* long double support under SunOS/gcc is pretty much non-existent.
     You'll get the odd bus error in library routines like printf() */
#else
  printf ("%Lf, %Lf, %Lf, %Lf\n", ld, ldblit(f), ld - ldblit(f), LDBL_EPSILON);
#endif

  /* These are not always the same!! Check for a reasonable delta */
  original = ldblit(f);
  if (((ld > original) ? (ld - original) : (original - ld)) < LDBL_EPSILON)
    puts("long double return value tests ok!");
  else
    CHECK(0);

  exit(0);
}

