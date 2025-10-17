/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 17, 2023.
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
#include <complex.h>

static void cls_ret_complex_fn(ffi_cif* cif __UNUSED__, void* resp, void** args,
			      void* userdata __UNUSED__)
 {
   _Complex T_C_TYPE *pa;
   _Complex T_C_TYPE *pr;
   pa = (_Complex T_C_TYPE *)args[0];
   pr = (_Complex T_C_TYPE *)resp;
   *pr = *pa;

   printf("%.6f,%.6fi: %.6f,%.6fi\n",
	  T_CONV creal (*pa), T_CONV cimag (*pa),
	  T_CONV creal (*pr), T_CONV cimag (*pr));
 }
typedef _Complex T_C_TYPE (*cls_ret_complex)(_Complex T_C_TYPE);

int main (void)
{
  ffi_cif cif;
  void *code;
  ffi_closure *pcl = ffi_closure_alloc(sizeof(ffi_closure), &code);
  ffi_type * cl_arg_types[2];
  _Complex T_C_TYPE res;

  cl_arg_types[0] = &T_FFI_TYPE;
  cl_arg_types[1] = NULL;

  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 1,
		     &T_FFI_TYPE, cl_arg_types) == FFI_OK);

  CHECK(ffi_prep_closure_loc(pcl, &cif, cls_ret_complex_fn, NULL, code)  == FFI_OK);

  res = (*((cls_ret_complex)code))(0.125 + 128.0 * I);
  printf("res: %.6f,%.6fi\n", T_CONV creal (res), T_CONV cimag (res));
  CHECK (res == (0.125 + 128.0 * I));

  exit(0);
}
