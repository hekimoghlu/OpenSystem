/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 4, 2024.
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

typedef struct cls_struct_18byte {
  double a;
  unsigned char b;
  unsigned char c;
  double d;
} cls_struct_18byte;

cls_struct_18byte cls_struct_18byte_fn(struct cls_struct_18byte a1,
			    struct cls_struct_18byte a2)
{
  struct cls_struct_18byte result;

  result.a = a1.a + a2.a;
  result.b = a1.b + a2.b;
  result.c = a1.c + a2.c;
  result.d = a1.d + a2.d;


  printf("%g %d %d %g %g %d %d %g: %g %d %d %g\n", a1.a, a1.b, a1.c, a1.d,
	 a2.a, a2.b, a2.c, a2.d,
	 result.a, result.b, result.c, result.d);
  return result;
}

static void
cls_struct_18byte_gn(ffi_cif* cif, void* resp, void** args, void* userdata)
{
  struct cls_struct_18byte a1, a2;

  a1 = *(struct cls_struct_18byte*)(args[0]);
  a2 = *(struct cls_struct_18byte*)(args[1]);

  *(cls_struct_18byte*)resp = cls_struct_18byte_fn(a1, a2);
}

int main (void)
{
  ffi_cif cif;
#ifndef USING_MMAP
  static ffi_closure cl;
#endif
  ffi_closure *pcl;
  void* args_dbl[3];
  ffi_type* cls_struct_fields[5];
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

  struct cls_struct_18byte g_dbl = { 1.0, 127, 126, 3.0 };
  struct cls_struct_18byte f_dbl = { 4.0, 125, 124, 5.0 };
  struct cls_struct_18byte res_dbl;

  cls_struct_fields[0] = &ffi_type_double;
  cls_struct_fields[1] = &ffi_type_uchar;
  cls_struct_fields[2] = &ffi_type_uchar;
  cls_struct_fields[3] = &ffi_type_double;
  cls_struct_fields[4] = NULL;

  dbl_arg_types[0] = &cls_struct_type;
  dbl_arg_types[1] = &cls_struct_type;
  dbl_arg_types[2] = NULL;

  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 2, &cls_struct_type,
		     dbl_arg_types) == FFI_OK);

  args_dbl[0] = &g_dbl;
  args_dbl[1] = &f_dbl;
  args_dbl[2] = NULL;

  ffi_call(&cif, FFI_FN(cls_struct_18byte_fn), &res_dbl, args_dbl);
  /* { dg-output "1 127 126 3 4 125 124 5: 5 252 250 8" } */
  printf("res: %g %d %d %g\n", res_dbl.a, res_dbl.b, res_dbl.c, res_dbl.d);
  /* { dg-output "\nres: 5 252 250 8" } */

  CHECK(ffi_prep_closure(pcl, &cif, cls_struct_18byte_gn, NULL) == FFI_OK);

  res_dbl = ((cls_struct_18byte(*)(cls_struct_18byte, cls_struct_18byte))(pcl))(g_dbl, f_dbl);
  /* { dg-output "\n1 127 126 3 4 125 124 5: 5 252 250 8" } */
  printf("res: %g %d %d %g\n", res_dbl.a, res_dbl.b, res_dbl.c, res_dbl.d);
  /* { dg-output "\nres: 5 252 250 8" } */

  exit(0);
}
