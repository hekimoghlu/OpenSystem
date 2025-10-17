/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 7, 2025.
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

typedef struct cls_struct_6byte {
  unsigned char a;
  unsigned char b;
  unsigned char c;
  unsigned char d;
  unsigned char e;
  unsigned char f;
} cls_struct_6byte;

static cls_struct_6byte cls_struct_6byte_fn(struct cls_struct_6byte a1,
			    struct cls_struct_6byte a2)
{
  struct cls_struct_6byte result;

  result.a = a1.a + a2.a;
  result.b = a1.b + a2.b;
  result.c = a1.c + a2.c;
  result.d = a1.d + a2.d;
  result.e = a1.e + a2.e;
  result.f = a1.f + a2.f;

  printf("%d %d %d %d %d %d %d %d %d %d %d %d: %d %d %d %d %d %d\n",
	 a1.a, a1.b, a1.c, a1.d, a1.e, a1.f,
	 a2.a, a2.b, a2.c, a2.d, a2.e, a2.f,
	 result.a, result.b, result.c, result.d, result.e, result.f);

  return  result;
}

static void
cls_struct_6byte_gn(ffi_cif* cif __UNUSED__, void* resp, void** args,
		    void* userdata __UNUSED__)
{

  struct cls_struct_6byte a1, a2;

  a1 = *(struct cls_struct_6byte*)(args[0]);
  a2 = *(struct cls_struct_6byte*)(args[1]);

  *(cls_struct_6byte*)resp = cls_struct_6byte_fn(a1, a2);
}

int main (void)
{
  ffi_cif cif;
  void *code;
  ffi_closure *pcl = ffi_closure_alloc(sizeof(ffi_closure), &code);
  void* args_dbl[5];
  ffi_type* cls_struct_fields[7];
  ffi_type cls_struct_type;
  ffi_type* dbl_arg_types[5];

  struct cls_struct_6byte g_dbl = { 127, 120, 1, 3, 4, 5 };
  struct cls_struct_6byte f_dbl = { 12, 128, 9, 3, 4, 5 };
  struct cls_struct_6byte res_dbl = { 0, 0, 0, 0, 0, 0 };

  cls_struct_type.size = 0;
  cls_struct_type.alignment = 0;
  cls_struct_type.type = FFI_TYPE_STRUCT;
  cls_struct_type.elements = cls_struct_fields;

  cls_struct_fields[0] = &ffi_type_uchar;
  cls_struct_fields[1] = &ffi_type_uchar;
  cls_struct_fields[2] = &ffi_type_uchar;
  cls_struct_fields[3] = &ffi_type_uchar;
  cls_struct_fields[4] = &ffi_type_uchar;
  cls_struct_fields[5] = &ffi_type_uchar;
  cls_struct_fields[6] = NULL;

  dbl_arg_types[0] = &cls_struct_type;
  dbl_arg_types[1] = &cls_struct_type;
  dbl_arg_types[2] = NULL;

  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 2, &cls_struct_type,
		     dbl_arg_types) == FFI_OK);

  args_dbl[0] = &g_dbl;
  args_dbl[1] = &f_dbl;
  args_dbl[2] = NULL;

  ffi_call(&cif, FFI_FN(cls_struct_6byte_fn), &res_dbl, args_dbl);
  /* { dg-output "127 120 1 3 4 5 12 128 9 3 4 5: 139 248 10 6 8 10" } */
  printf("res: %d %d %d %d %d %d\n", res_dbl.a, res_dbl.b, res_dbl.c,
	 res_dbl.d, res_dbl.e, res_dbl.f);
  /* { dg-output "\nres: 139 248 10 6 8 10" } */

  res_dbl.a = 0;
  res_dbl.b = 0;
  res_dbl.c = 0;
  res_dbl.d = 0;
  res_dbl.e = 0;
  res_dbl.f = 0;

  CHECK(ffi_prep_closure_loc(pcl, &cif, cls_struct_6byte_gn, NULL, code) == FFI_OK);

  res_dbl = ((cls_struct_6byte(*)(cls_struct_6byte, cls_struct_6byte))(code))(g_dbl, f_dbl);
  /* { dg-output "\n127 120 1 3 4 5 12 128 9 3 4 5: 139 248 10 6 8 10" } */
  printf("res: %d %d %d %d %d %d\n", res_dbl.a, res_dbl.b, res_dbl.c,
	 res_dbl.d, res_dbl.e, res_dbl.f);
  /* { dg-output "\nres: 139 248 10 6 8 10" } */

  exit(0);
}

