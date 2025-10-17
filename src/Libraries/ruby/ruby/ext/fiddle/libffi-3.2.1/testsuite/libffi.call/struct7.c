/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
typedef struct
{
  float f1;
  float f2;
  double d;
} test_structure_7;

static test_structure_7 ABI_ATTR struct7 (test_structure_7 ts)
{
  ts.f1 += 1;
  ts.f2 += 1;
  ts.d += 1;

  return ts;
}

int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  ffi_type ts7_type;
  ffi_type *ts7_type_elements[4];

  test_structure_7 ts7_arg;

  /* This is a hack to get a properly aligned result buffer */
  test_structure_7 *ts7_result =
    (test_structure_7 *) malloc (sizeof(test_structure_7));

  ts7_type.size = 0;
  ts7_type.alignment = 0;
  ts7_type.type = FFI_TYPE_STRUCT;
  ts7_type.elements = ts7_type_elements;
  ts7_type_elements[0] = &ffi_type_float;
  ts7_type_elements[1] = &ffi_type_float;
  ts7_type_elements[2] = &ffi_type_double;
  ts7_type_elements[3] = NULL;

  args[0] = &ts7_type;
  values[0] = &ts7_arg;
  
  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, ABI_NUM, 1, &ts7_type, args) == FFI_OK);
  
  ts7_arg.f1 = 5.55f;
  ts7_arg.f2 = 55.5f;
  ts7_arg.d = 6.66;

  printf ("%g\n", ts7_arg.f1);
  printf ("%g\n", ts7_arg.f2);
  printf ("%g\n", ts7_arg.d);
  
  ffi_call(&cif, FFI_FN(struct7), ts7_result, values);

  printf ("%g\n", ts7_result->f1);
  printf ("%g\n", ts7_result->f2);
  printf ("%g\n", ts7_result->d);
  
  CHECK(ts7_result->f1 == 5.55f + 1);
  CHECK(ts7_result->f2 == 55.5f + 1);
  CHECK(ts7_result->d == 6.66 + 1);
  
  free (ts7_result);
  exit(0);
}
