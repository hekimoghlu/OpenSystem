/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 20, 2023.
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
  int si;
} test_structure_3;

static test_structure_3 struct3(test_structure_3 ts)
{
  ts.si = -(ts.si*2);

  return ts;
}

int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  int compare_value;
  ffi_type ts3_type;
  ffi_type *ts3_type_elements[2];
  ts3_type.size = 0;
  ts3_type.alignment = 0;
  ts3_type.type = FFI_TYPE_STRUCT;
  ts3_type.elements = ts3_type_elements;
  ts3_type_elements[0] = &ffi_type_sint;
  ts3_type_elements[1] = NULL;

  test_structure_3 ts3_arg;
  test_structure_3 *ts3_result = 
    (test_structure_3 *) malloc (sizeof(test_structure_3));
  
  args[0] = &ts3_type;
  values[0] = &ts3_arg;
  
  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 1, 
		     &ts3_type, args) == FFI_OK);
  
  ts3_arg.si = -123;
  compare_value = ts3_arg.si;
  
  ffi_call(&cif, FFI_FN(struct3), ts3_result, values);
  
  printf ("%d %d\n", ts3_result->si, -(compare_value*2));
  
  CHECK(ts3_result->si == -(compare_value*2));
 
  free (ts3_result);
  exit(0);
}
