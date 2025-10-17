/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 2, 2022.
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
  float f3;
  float f4;
} test_structure_8;

static test_structure_8 ABI_ATTR struct8 (test_structure_8 ts)
{
  ts.f1 += 1;
  ts.f2 += 1;
  ts.f3 += 1;
  ts.f4 += 1;

  return ts;
}

int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  ffi_type ts8_type;
  ffi_type *ts8_type_elements[5];

  test_structure_8 ts8_arg;

  /* This is a hack to get a properly aligned result buffer */
  test_structure_8 *ts8_result =
    (test_structure_8 *) malloc (sizeof(test_structure_8));

  ts8_type.size = 0;
  ts8_type.alignment = 0;
  ts8_type.type = FFI_TYPE_STRUCT;
  ts8_type.elements = ts8_type_elements;
  ts8_type_elements[0] = &ffi_type_float;
  ts8_type_elements[1] = &ffi_type_float;
  ts8_type_elements[2] = &ffi_type_float;
  ts8_type_elements[3] = &ffi_type_float;
  ts8_type_elements[4] = NULL;

  args[0] = &ts8_type;
  values[0] = &ts8_arg;
  
  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, ABI_NUM, 1, &ts8_type, args) == FFI_OK);
  
  ts8_arg.f1 = 5.55f;
  ts8_arg.f2 = 55.5f;
  ts8_arg.f3 = -5.55f;
  ts8_arg.f4 = -55.5f;

  printf ("%g\n", ts8_arg.f1);
  printf ("%g\n", ts8_arg.f2);
  printf ("%g\n", ts8_arg.f3);
  printf ("%g\n", ts8_arg.f4);
  
  ffi_call(&cif, FFI_FN(struct8), ts8_result, values);

  printf ("%g\n", ts8_result->f1);
  printf ("%g\n", ts8_result->f2);
  printf ("%g\n", ts8_result->f3);
  printf ("%g\n", ts8_result->f4);
  
  CHECK(ts8_result->f1 == 5.55f + 1);
  CHECK(ts8_result->f2 == 55.5f + 1);
  CHECK(ts8_result->f3 == -5.55f + 1);
  CHECK(ts8_result->f4 == -55.5f + 1);
  
  free (ts8_result);
  exit(0);
}

