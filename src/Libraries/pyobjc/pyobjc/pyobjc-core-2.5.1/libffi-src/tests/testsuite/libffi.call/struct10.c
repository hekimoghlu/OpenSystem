/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 19, 2023.
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

typedef struct {
	float x;
	float y;
} point;

typedef struct
{
  point l;
  point r;
} test_structure_10;

static test_structure_10 struct10 (test_structure_10 ts)
{
  ts.r.x += 1.0;
  ts.r.y += 1.5;
  ts.l.x += 2.0;
  ts.l.y += 2.5;

  return ts;
}

static test_structure_10 struct10b (test_structure_10 ts)
{
	return struct10(ts);
}

int main (void)
{
  ffi_cif cif;
  ffi_type *args[MAX_ARGS];
  void *values[MAX_ARGS];
  ffi_type ts10_type;
  ffi_type *ts10_type_elements[3];
  ffi_type ts10p_type;
  ffi_type *ts10p_type_elements[3];

  ts10p_type.size = 0;
  ts10p_type.alignment = 0;
  ts10p_type.type = FFI_TYPE_STRUCT;
  ts10p_type.elements = ts10p_type_elements;
  ts10p_type_elements[0] = &ffi_type_float;
  ts10p_type_elements[1] = &ffi_type_float;
  ts10p_type_elements[2] = NULL;

  ts10_type.size = 0;
  ts10_type.alignment = 0;
  ts10_type.type = FFI_TYPE_STRUCT;
  ts10_type.elements = ts10_type_elements;
  ts10_type_elements[0] = &ts10p_type;
  ts10_type_elements[1] = &ts10p_type;
  ts10_type_elements[2] = NULL;

  test_structure_10 ts10_arg;
  
  /* This is a hack to get a properly aligned result buffer */
  test_structure_10 *ts10_result = 
    (test_structure_10 *) malloc (sizeof(test_structure_10));
  
  args[0] = &ts10_type;
  values[0] = &ts10_arg;
  
  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 1, &ts10_type, args) == FFI_OK);
  
  ts10_arg.r.x = 1.44;
  ts10_arg.r.y = 2.44;
  ts10_arg.l.x = 3.44;
  ts10_arg.l.y = 4.44;
  
  printf ("%g\n", ts10_arg.r.x);
  printf ("%g\n", ts10_arg.r.y);
  printf ("%g\n", ts10_arg.l.x);
  printf ("%g\n", ts10_arg.l.y);
  
  ffi_call(&cif, FFI_FN(struct10b), ts10_result, values);

  printf ("%g\n", ts10_result->r.x);
  printf ("%g\n", ts10_result->r.y);
  printf ("%g\n", ts10_result->l.x);
  printf ("%g\n", ts10_result->l.y);

  CHECK(ts10_result->r.x == 1.44f + 1.0);
  CHECK(ts10_result->r.y == 2.44f + 1.5);
  CHECK(ts10_result->l.x == 3.44f + 2.0);
  CHECK(ts10_result->l.y == 4.44f + 2.5);

  free (ts10_result);
  exit(0);
}
