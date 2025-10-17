/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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
#include "ffitest.h"

typedef struct
{
  unsigned char uc;
  double d;
  unsigned int ui;
} test_structure_1;

static test_structure_1 struct1(test_structure_1 ts)
{
  ts.uc++;
  ts.d--;
  ts.ui++;

  return ts;
}

int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  ffi_type ts1_type;
  ffi_type *ts1_type_elements[4];

  memset(&cif, 1, sizeof(cif));
  ts1_type.size = 0;
  ts1_type.alignment = 0;
  ts1_type.type = FFI_TYPE_STRUCT;
  ts1_type.elements = ts1_type_elements;
  ts1_type_elements[0] = &ffi_type_uchar;
  ts1_type_elements[1] = &ffi_type_double;
  ts1_type_elements[2] = &ffi_type_uint;
  ts1_type_elements[3] = NULL;

  test_structure_1 ts1_arg;
  /* This is a hack to get a properly aligned result buffer */
  test_structure_1 *ts1_result =
    (test_structure_1 *) malloc (sizeof(test_structure_1));

  args[0] = &ts1_type;
  values[0] = &ts1_arg;

  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 1,
		     &ts1_type, args) == FFI_OK);

  ts1_arg.uc = '\x01';
  ts1_arg.d = 3.14159;
  ts1_arg.ui = 555;

  ffi_call(&cif, FFI_FN(struct1), ts1_result, values);

  CHECK(ts1_result->ui == 556);
  CHECK(ts1_result->d == 3.14159 - 1);

  free (ts1_result);
  exit(0);
}
