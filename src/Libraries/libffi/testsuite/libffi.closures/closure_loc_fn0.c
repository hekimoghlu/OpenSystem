/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 15, 2025.
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

static void
closure_loc_test_fn0(ffi_cif* cif __UNUSED__, void* resp, void** args,
		 void* userdata)
{
  *(ffi_arg*)resp =
    (int)*(unsigned long long *)args[0] + (int)(*(int *)args[1]) +
    (int)(*(unsigned long long *)args[2]) + (int)*(int *)args[3] +
    (int)(*(signed short *)args[4]) +
    (int)(*(unsigned long long *)args[5]) +
    (int)*(int *)args[6] + (int)(*(int *)args[7]) +
    (int)(*(double *)args[8]) + (int)*(int *)args[9] +
    (int)(*(int *)args[10]) + (int)(*(float *)args[11]) +
    (int)*(int *)args[12] + (int)(*(int *)args[13]) +
    (int)(*(int *)args[14]) +  *(int *)args[15] + (intptr_t)userdata;

  printf("%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d: %d\n",
	 (int)*(unsigned long long *)args[0], (int)(*(int *)args[1]),
	 (int)(*(unsigned long long *)args[2]),
	 (int)*(int *)args[3], (int)(*(signed short *)args[4]),
	 (int)(*(unsigned long long *)args[5]),
	 (int)*(int *)args[6], (int)(*(int *)args[7]),
	 (int)(*(double *)args[8]), (int)*(int *)args[9],
	 (int)(*(int *)args[10]), (int)(*(float *)args[11]),
	 (int)*(int *)args[12], (int)(*(int *)args[13]),
	 (int)(*(int *)args[14]),*(int *)args[15],
	 (int)(intptr_t)userdata, (int)*(ffi_arg *)resp);

}

typedef int (*closure_loc_test_type0)(unsigned long long, int, unsigned long long,
				  int, signed short, unsigned long long, int,
				  int, double, int, int, float, int, int,
				  int, int);

int main (void)
{
  ffi_cif cif;
  ffi_closure *pcl;
  ffi_type * cl_arg_types[17];
  int res;
  void *codeloc;

  cl_arg_types[0] = &ffi_type_uint64;
  cl_arg_types[1] = &ffi_type_sint;
  cl_arg_types[2] = &ffi_type_uint64;
  cl_arg_types[3] = &ffi_type_sint;
  cl_arg_types[4] = &ffi_type_sshort;
  cl_arg_types[5] = &ffi_type_uint64;
  cl_arg_types[6] = &ffi_type_sint;
  cl_arg_types[7] = &ffi_type_sint;
  cl_arg_types[8] = &ffi_type_double;
  cl_arg_types[9] = &ffi_type_sint;
  cl_arg_types[10] = &ffi_type_sint;
  cl_arg_types[11] = &ffi_type_float;
  cl_arg_types[12] = &ffi_type_sint;
  cl_arg_types[13] = &ffi_type_sint;
  cl_arg_types[14] = &ffi_type_sint;
  cl_arg_types[15] = &ffi_type_sint;
  cl_arg_types[16] = NULL;

  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 16,
		     &ffi_type_sint, cl_arg_types) == FFI_OK);

  pcl = ffi_closure_alloc(sizeof(ffi_closure), &codeloc);
  CHECK(pcl != NULL);
  CHECK(codeloc != NULL);

  CHECK(ffi_prep_closure_loc(pcl, &cif, closure_loc_test_fn0,
			 (void *) 3 /* userdata */, codeloc) == FFI_OK);
  
#ifndef FFI_EXEC_STATIC_TRAMP
  /* With static trampolines, the codeloc does not point to closure */
  CHECK(memcmp(pcl, codeloc, sizeof(*pcl)) == 0);
#endif

  res = (*((closure_loc_test_type0)codeloc))
    (1LL, 2, 3LL, 4, 127, 429LL, 7, 8, 9.5, 10, 11, 12, 13,
     19, 21, 1);
  /* { dg-output "1 2 3 4 127 429 7 8 9 10 11 12 13 19 21 1 3: 680" } */
  printf("res: %d\n",res);
  /* { dg-output "\nres: 680" } */
     exit(0);
}

