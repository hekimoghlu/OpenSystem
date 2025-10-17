/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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

static void cls_ret_sint_fn(ffi_cif* cif,void* resp,void** args,
			     void* userdata)
{
  *(ffi_arg*)resp = *(signed int *)args[0];
  printf("%d: %d\n",*(signed int *)args[0],
	 *(ffi_arg*)resp);
}
typedef signed int (*cls_ret_sint)(signed int);

int main (void)
{
  ffi_cif cif;
#ifndef USING_MMAP
  static ffi_closure cl;
#endif
  ffi_closure *pcl;
  ffi_type * cl_arg_types[2];
  signed int res;

#ifdef USING_MMAP
  pcl = allocate_mmap (sizeof(ffi_closure));
#else
  pcl = &cl;
#endif

  cl_arg_types[0] = &ffi_type_sint32;
  cl_arg_types[1] = NULL;

  /* Initialize the cif */
  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 1,
		     &ffi_type_sint32, cl_arg_types) == FFI_OK);

  CHECK(ffi_prep_closure(pcl, &cif, cls_ret_sint_fn, NULL)  == FFI_OK);

  res = (*((cls_ret_sint)pcl))(65534);
  /* { dg-output "65534: 65534" } */
  printf("res: %d\n",res);
  /* { dg-output "\nres: 65534" } */

  exit(0);
}
