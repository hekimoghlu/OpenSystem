/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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
/* { dg-options "-Wno-format" { target alpha*-dec-osf* } } */
#include "ffitest.h"

static void cls_ret_ulonglong_fn(ffi_cif* cif __UNUSED__, void* resp,
				 void** args, void* userdata __UNUSED__)
{
  *(unsigned long long *)resp= 0xfffffffffffffffLL ^ *(unsigned long long *)args[0];

  printf("%" PRIuLL ": %" PRIuLL "\n",*(unsigned long long *)args[0],
	 *(unsigned long long *)(resp));
}
typedef unsigned long long (*cls_ret_ulonglong)(unsigned long long);

int main (void)
{
  ffi_cif cif;
  void *code;
  ffi_closure *pcl = ffi_closure_alloc(sizeof(ffi_closure), &code);
  ffi_type * cl_arg_types[2];
  unsigned long long res;

  cl_arg_types[0] = &ffi_type_uint64;
  cl_arg_types[1] = NULL;

  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 1,
		     &ffi_type_uint64, cl_arg_types) == FFI_OK);
  CHECK(ffi_prep_closure_loc(pcl, &cif, cls_ret_ulonglong_fn, NULL, code)  == FFI_OK);
  res = (*((cls_ret_ulonglong)code))(214LL);
  /* { dg-output "214: 1152921504606846761" } */
  printf("res: %" PRIdLL "\n", res);
  /* { dg-output "\nres: 1152921504606846761" } */

  res = (*((cls_ret_ulonglong)code))(9223372035854775808LL);
  /* { dg-output "\n9223372035854775808: 8070450533247928831" } */
  printf("res: %" PRIdLL "\n", res);
  /* { dg-output "\nres: 8070450533247928831" } */

  exit(0);
}
