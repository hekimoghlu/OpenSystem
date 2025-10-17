/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 17, 2024.
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
  double d1;
  double d2;
} test_structure_2;

static test_structure_2 ABI_ATTR struct2(test_structure_2 ts)
{
  ts.d1--;
  ts.d2--;

  return ts;
}

int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  test_structure_2 ts2_arg;
  ffi_type ts2_type;
  ffi_type *ts2_type_elements[3];

  /* This is a hack to get a properly aligned result buffer */
  test_structure_2 *ts2_result =
    (test_structure_2 *) malloc (sizeof(test_structure_2));

  ts2_type.size = 0;
  ts2_type.alignment = 0;
  ts2_type.type = FFI_TYPE_STRUCT;
  ts2_type.elements = ts2_type_elements;
  ts2_type_elements[0] = &ffi_type_double;
  ts2_type_elements[1] = &ffi_type_double;
  ts2_type_elements[2] = NULL;

  args[0] = &ts2_type;
  values[0] = &ts2_arg;
  
  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, ABI_NUM, 1, &ts2_type, args) == FFI_OK);
  
  ts2_arg.d1 = 5.55;
  ts2_arg.d2 = 6.66;
  
  printf ("%g\n", ts2_arg.d1);
  printf ("%g\n", ts2_arg.d2);
  
  ffi_call(&cif, FFI_FN(struct2), ts2_result, values);
  
  printf ("%g\n", ts2_result->d1);
  printf ("%g\n", ts2_result->d2);
  
  CHECK(ts2_result->d1 == 5.55 - 1);
  CHECK(ts2_result->d2 == 6.66 - 1);
  
  free (ts2_result);
  exit(0);
}

