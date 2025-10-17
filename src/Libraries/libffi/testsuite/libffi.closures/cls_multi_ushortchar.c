/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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

static unsigned short test_func_fn(unsigned char a1, unsigned short a2,
			    unsigned char a3, unsigned short a4)
{
  unsigned short result;

  result = a1 + a2 + a3 + a4;

  printf("%d %d %d %d: %d\n", a1, a2, a3, a4, result);

  return result;

}

static void test_func_gn(ffi_cif *cif __UNUSED__, void *rval, void **avals,
			 void *data __UNUSED__)
{
  unsigned char a1, a3;
  unsigned short a2, a4;

  a1 = *(unsigned char *)avals[0];
  a2 = *(unsigned short *)avals[1];
  a3 = *(unsigned char *)avals[2];
  a4 = *(unsigned short *)avals[3];

  *(ffi_arg *)rval = test_func_fn(a1, a2, a3, a4);

}

typedef unsigned short (*test_type)(unsigned char, unsigned short,
				   unsigned char, unsigned short);

int main (void)
{
  ffi_cif cif;
  void *code;
  ffi_closure *pcl = ffi_closure_alloc(sizeof(ffi_closure), &code);
  void * args_dbl[5];
  ffi_type * cl_arg_types[5];
  ffi_arg res_call;
  unsigned char a, c;
  unsigned short b, d, res_closure;

  a = 1;
  b = 2;
  c = 127;
  d = 128;

  args_dbl[0] = &a;
  args_dbl[1] = &b;
  args_dbl[2] = &c;
  args_dbl[3] = &d;
  args_dbl[4] = NULL;

  cl_arg_types[0] = &ffi_type_uchar;
  cl_arg_types[1] = &ffi_type_ushort;
  cl_arg_types[2] = &ffi_type_uchar;
  cl_arg_types[3] = &ffi_type_ushort;
  cl_arg_types[4] = NULL;

  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 4,
		     &ffi_type_ushort, cl_arg_types) == FFI_OK);

  ffi_call(&cif, FFI_FN(test_func_fn), &res_call, args_dbl);
  /* { dg-output "1 2 127 128: 258" } */
  printf("res: %d\n", (unsigned short)res_call);
  /* { dg-output "\nres: 258" } */

  CHECK(ffi_prep_closure_loc(pcl, &cif, test_func_gn, NULL, code)  == FFI_OK);

  res_closure = (*((test_type)code))(1, 2, 127, 128);
  /* { dg-output "\n1 2 127 128: 258" } */
  printf("res: %d\n", res_closure);
  /* { dg-output "\nres: 258" } */

  exit(0);
}

