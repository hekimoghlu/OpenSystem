/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 26, 2024.
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
  unsigned ui1;
  unsigned ui2;
  unsigned ui3;
} test_structure_4;

static test_structure_4 ABI_ATTR struct4(test_structure_4 ts)
{
  ts.ui3 = ts.ui1 * ts.ui2 * ts.ui3;

  return ts;
}

int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  ffi_type ts4_type;
  ffi_type *ts4_type_elements[4];  

  test_structure_4 ts4_arg;

  /* This is a hack to get a properly aligned result buffer */
  test_structure_4 *ts4_result =
    (test_structure_4 *) malloc (sizeof(test_structure_4));

  ts4_type.size = 0;
  ts4_type.alignment = 0;
  ts4_type.type = FFI_TYPE_STRUCT;
  ts4_type.elements = ts4_type_elements;
  ts4_type_elements[0] = &ffi_type_uint;
  ts4_type_elements[1] = &ffi_type_uint;
  ts4_type_elements[2] = &ffi_type_uint;
  ts4_type_elements[3] = NULL;

  args[0] = &ts4_type;
  values[0] = &ts4_arg;
  
  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, ABI_NUM, 1, &ts4_type, args) == FFI_OK);
  
  ts4_arg.ui1 = 2;
  ts4_arg.ui2 = 3;
  ts4_arg.ui3 = 4;
  
  ffi_call (&cif, FFI_FN(struct4), ts4_result, values);
  
  CHECK(ts4_result->ui3 == 2U * 3U * 4U);
 
  
  free (ts4_result);
  exit(0);
}
