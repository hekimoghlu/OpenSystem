/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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

typedef struct cls_struct_9byte {
  int a;
  double b;
} cls_struct_9byte;

cls_struct_9byte cls_struct_9byte_fn(struct cls_struct_9byte b1,
			    struct cls_struct_9byte b2)
{
  struct cls_struct_9byte result;

  result.a = b1.a + b2.a;
  result.b = b1.b + b2.b;

  printf("%d %g %d %g: %d %g\n", b1.a, b1.b,  b2.a, b2.b,
	 result.a, result.b);

  return result;
}

static void cls_struct_9byte_gn(ffi_cif* cif, void* resp, void** args,
				void* userdata)
{
  struct cls_struct_9byte b1, b2;

  b1 = *(struct cls_struct_9byte*)(args[0]);
  b2 = *(struct cls_struct_9byte*)(args[1]);

  *(cls_struct_9byte*)resp = cls_struct_9byte_fn(b1, b2);
}

int main (void)
{
  ffi_cif cif;
#ifndef USING_MMAP
  static ffi_closure cl;
#endif
  ffi_closure *pcl;
  void* args_dbl[3];
  ffi_type* cls_struct_fields[3];
  ffi_type cls_struct_type;
  ffi_type* dbl_arg_types[3];

#ifdef USING_MMAP
  pcl = allocate_mmap (sizeof(ffi_closure));
#else
  pcl = &cl;
#endif

  cls_struct_type.size = 0;
  cls_struct_type.alignment = 0;
  cls_struct_type.type = FFI_TYPE_STRUCT;
  cls_struct_type.elements = cls_struct_fields;

  struct cls_struct_9byte h_dbl = { 7, 8.0};
  struct cls_struct_9byte j_dbl = { 1, 9.0};
  struct cls_struct_9byte res_dbl;

  cls_struct_fields[0] = &ffi_type_uint32;
  cls_struct_fields[1] = &ffi_type_double;
  cls_struct_fields[2] = NULL;

  dbl_arg_types[0] = &cls_struct_type;
  dbl_arg_types[1] = &cls_struct_type;
  dbl_arg_types[2] = NULL;

  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 2, &cls_struct_type,
		     dbl_arg_types) == FFI_OK);

  args_dbl[0] = &h_dbl;
  args_dbl[1] = &j_dbl;
  args_dbl[2] = NULL;

  ffi_call(&cif, FFI_FN(cls_struct_9byte_fn), &res_dbl, args_dbl);
  /* { dg-output "7 8 1 9: 8 17" } */
  printf("res: %d %g\n", res_dbl.a, res_dbl.b);
  /* { dg-output "\nres: 8 17" } */

  CHECK(ffi_prep_closure(pcl, &cif, cls_struct_9byte_gn, NULL) == FFI_OK);

  res_dbl = ((cls_struct_9byte(*)(cls_struct_9byte, cls_struct_9byte))(pcl))(h_dbl, j_dbl);
  /* { dg-output "\n7 8 1 9: 8 17" } */
  printf("res: %d %g\n", res_dbl.a, res_dbl.b);
  /* { dg-output "\nres: 8 17" } */

  exit(0);
}
