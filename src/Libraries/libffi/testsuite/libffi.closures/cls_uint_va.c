/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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

typedef unsigned int T;

static void cls_ret_T_fn(ffi_cif* cif __UNUSED__, void* resp, void** args,
			 void* userdata __UNUSED__)
 {
   *(ffi_arg *)resp = *(T *)args[0];

   printf("%d: %d %d\n", (int)*(ffi_arg *)resp, *(T *)args[0], *(T *)args[1]);
 }

typedef T (*cls_ret_T)(T, ...);

int main (void)
{
  ffi_cif cif;
  void *code;
  ffi_closure *pcl = ffi_closure_alloc(sizeof(ffi_closure), &code);
  ffi_type * cl_arg_types[3];
  T res;

  cl_arg_types[0] = &ffi_type_uint;
  cl_arg_types[1] = &ffi_type_uint;
  cl_arg_types[2] = NULL;

  /* Initialize the cif */
  CHECK(ffi_prep_cif_var(&cif, FFI_DEFAULT_ABI, 1, 2,
			 &ffi_type_uint, cl_arg_types) == FFI_OK);

  CHECK(ffi_prep_closure_loc(pcl, &cif, cls_ret_T_fn, NULL, code)  == FFI_OK);
  res = ((((cls_ret_T)code)(67, 4)));
  /* { dg-output "67: 67 4" } */
  printf("res: %d\n", res);
  /* { dg-output "\nres: 67" } */
  exit(0);
}

