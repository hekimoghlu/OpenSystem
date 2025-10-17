/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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
#include <stddef.h>

struct test_1
{
  char c;
  float f;
  char c2;
  int i;
};

int
main (void)
{
  ffi_type test_1_type;
  ffi_type *test_1_elements[5];
  size_t test_1_offsets[4];

  test_1_elements[0] = &ffi_type_schar;
  test_1_elements[1] = &ffi_type_float;
  test_1_elements[2] = &ffi_type_schar;
  test_1_elements[3] = &ffi_type_sint;
  test_1_elements[4] = NULL;

  test_1_type.size = 0;
  test_1_type.alignment = 0;
  test_1_type.type = FFI_TYPE_STRUCT;
  test_1_type.elements = test_1_elements;

  CHECK (ffi_get_struct_offsets (FFI_DEFAULT_ABI, &test_1_type, test_1_offsets)
	 == FFI_OK);
  CHECK (test_1_type.size == sizeof (struct test_1));
  CHECK (offsetof (struct test_1, c) == test_1_offsets[0]);
  CHECK (offsetof (struct test_1, f) == test_1_offsets[1]);
  CHECK (offsetof (struct test_1, c2) == test_1_offsets[2]);
  CHECK (offsetof (struct test_1, i) == test_1_offsets[3]);

  return 0;
}

