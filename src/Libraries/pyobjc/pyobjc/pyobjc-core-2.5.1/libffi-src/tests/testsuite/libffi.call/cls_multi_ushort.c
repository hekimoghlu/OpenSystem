/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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
/* { dg-do run { xfail mips*-*-* arm*-*-* strongarm*-*-* xscale*-*-* } } */
#include "ffitest.h"

unsigned short test_func_fn(unsigned short a1, unsigned short a2)
{
  unsigned short result;

  result = a1 + a2;

  printf("%d %d: %d\n", a1, a2, result);

  return result;

}

static void test_func_gn(ffi_cif *cif, void *rval, void **avals, void *data)
{
  unsigned short a1, a2;

  a1 = *(unsigned short *)avals[0];
  a2 = *(unsigned short *)avals[1];

  *(ffi_arg *)rval = test_func_fn(a1, a2);

}

typedef unsigned short (*test_type)(unsigned short, unsigned short);

int main (void)
{
  ffi_cif cif;
#ifndef USING_MMAP
  static ffi_closure cl;
#endif
  ffi_closure *pcl;
  void * args_dbl[3];
  ffi_type * cl_arg_types[3];
  ffi_arg res_call;
  unsigned short a, b, res_closure;

#ifdef USING_MMAP
  pcl = allocate_mmap (sizeof(ffi_closure));
#else
  pcl = &cl;
#endif

  a = 2;
  b = 32765;

  args_dbl[0] = &a;
  args_dbl[1] = &b;
  args_dbl[2] = NULL;

  cl_arg_types[0] = &ffi_type_ushort;
  cl_arg_types[1] = &ffi_type_ushort;
  cl_arg_types[2] = NULL;

  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 2,
		     &ffi_type_ushort, cl_arg_types) == FFI_OK);

  ffi_call(&cif, FFI_FN(test_func_fn), &res_call, args_dbl);
  /* { dg-output "2 32765: 32767" } */
  printf("res: %d\n", res_call);
  /* { dg-output "\nres: 32767" } */

  CHECK(ffi_prep_closure(pcl, &cif, test_func_gn, NULL)  == FFI_OK);

  res_closure = (*((test_type)pcl))(2, 32765);
  /* { dg-output "\n2 32765: 32767" } */
  printf("res: %d\n", res_closure);
  /* { dg-output "\nres: 32767" } */

  exit(0);
}
