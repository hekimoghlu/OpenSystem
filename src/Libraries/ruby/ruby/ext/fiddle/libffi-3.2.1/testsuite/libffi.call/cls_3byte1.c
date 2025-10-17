/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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

typedef struct cls_struct_3byte {
  unsigned short a;
  unsigned char b;
} cls_struct_3byte;

cls_struct_3byte cls_struct_3byte_fn(struct cls_struct_3byte a1,
			    struct cls_struct_3byte a2)
{
  struct cls_struct_3byte result;

  result.a = a1.a + a2.a;
  result.b = a1.b + a2.b;

  printf("%d %d %d %d: %d %d\n", a1.a, a1.b, a2.a, a2.b, result.a, result.b);

  return  result;
}

static void
cls_struct_3byte_gn(ffi_cif* cif __UNUSED__, void* resp, void** args,
		    void* userdata __UNUSED__)
{

  struct cls_struct_3byte a1, a2;

  a1 = *(struct cls_struct_3byte*)(args[0]);
  a2 = *(struct cls_struct_3byte*)(args[1]);

  *(cls_struct_3byte*)resp = cls_struct_3byte_fn(a1, a2);
}

int main (void)
{
  ffi_cif cif;
  void *code;
  ffi_closure *pcl = ffi_closure_alloc(sizeof(ffi_closure), &code);
  void* args_dbl[5];
  ffi_type* cls_struct_fields[4];
  ffi_type cls_struct_type;
  ffi_type* dbl_arg_types[5];

  struct cls_struct_3byte g_dbl = { 12, 119 };
  struct cls_struct_3byte f_dbl = { 1, 15 };
  struct cls_struct_3byte res_dbl;

  cls_struct_type.size = 0;
  cls_struct_type.alignment = 0;
  cls_struct_type.type = FFI_TYPE_STRUCT;
  cls_struct_type.elements = cls_struct_fields;

  cls_struct_fields[0] = &ffi_type_ushort;
  cls_struct_fields[1] = &ffi_type_uchar;
  cls_struct_fields[2] = NULL;

  dbl_arg_types[0] = &cls_struct_type;
  dbl_arg_types[1] = &cls_struct_type;
  dbl_arg_types[2] = NULL;

  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 2, &cls_struct_type,
		     dbl_arg_types) == FFI_OK);

  args_dbl[0] = &g_dbl;
  args_dbl[1] = &f_dbl;
  args_dbl[2] = NULL;

  ffi_call(&cif, FFI_FN(cls_struct_3byte_fn), &res_dbl, args_dbl);
  /* { dg-output "12 119 1 15: 13 134" } */
  printf("res: %d %d\n", res_dbl.a, res_dbl.b);
  /* { dg-output "\nres: 13 134" } */

  CHECK(ffi_prep_closure_loc(pcl, &cif, cls_struct_3byte_gn, NULL, code) == FFI_OK);

  res_dbl = ((cls_struct_3byte(*)(cls_struct_3byte, cls_struct_3byte))(code))(g_dbl, f_dbl);
  /* { dg-output "\n12 119 1 15: 13 134" } */
  printf("res: %d %d\n", res_dbl.a, res_dbl.b);
  /* { dg-output "\nres: 13 134" } */

  exit(0);
}
