/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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

struct P {
	double x;
	double y;
};

static void
base_func(struct P p)
{
  printf("%.1f %.1f\n",  p.x, p.y);
}


static void
closure_test_fn6(ffi_cif* cif,void* resp,void** args, void* userdata)
{
   base_func(*(struct P*)args[0]);
}

typedef void (*closure_test_type0)(struct P);

int main (void)
{
  ffi_cif cif;
#ifndef USING_MMAP
  static ffi_closure cl;
#endif
  ffi_closure *pcl;
  ffi_type * cl_arg_types[1];
  int i, res;
#ifdef USING_MMAP
  pcl = allocate_mmap (sizeof(ffi_closure));
#else
  pcl = &cl;
#endif
  ffi_type ts1_type;
  ffi_type *ts1_type_elements[3];
  ts1_type.size = 0;
  ts1_type.alignment = 0;
  ts1_type.type = FFI_TYPE_STRUCT;
  ts1_type.elements = ts1_type_elements;
  ts1_type_elements[0] = &ffi_type_double;
  ts1_type_elements[1] = &ffi_type_double;
  ts1_type_elements[2] = NULL;

  cl_arg_types[0] = &ts1_type;

  struct P value = { 1.0, 2.0 };

  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 1,
		     &ffi_type_void, cl_arg_types) == FFI_OK);

  CHECK(ffi_prep_closure(pcl, &cif, closure_test_fn6,
			 (void *) 3 /* userdata */) == FFI_OK);

  (*((closure_test_type0)pcl))(value);
  /* { dg-output "1.0 2.0\n" } */

  base_func(value);
  /* { dg-output "1.0 2.0\n" } */

  exit(0);
}
