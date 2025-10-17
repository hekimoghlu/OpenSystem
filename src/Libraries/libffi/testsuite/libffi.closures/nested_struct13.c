/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 31, 2022.
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

typedef struct A {
  float a, b;
} A;

typedef struct B {
  float x;
  struct A y;
} B;

B B_fn(float b0, struct B b1)
{
  struct B result;

  result.x = b0 + b1.x;
  result.y.a = b0 + b1.y.a;
  result.y.b = b0 + b1.y.b;

  printf("%g %g %g %g: %g %g %g\n", b0, b1.x, b1.y.a, b1.y.b,
        result.x, result.y.a, result.y.b);

  return result;
}

static void
B_gn(ffi_cif* cif __UNUSED__, void* resp, void** args,
     void* userdata __UNUSED__)
{
  float b0;
  struct B b1;

  b0 = *(float*)(args[0]);
  b1 = *(struct B*)(args[1]);

  *(B*)resp = B_fn(b0, b1);
}

int main (void)
{
  ffi_cif cif;
  void *code;
  ffi_closure *pcl = ffi_closure_alloc(sizeof(ffi_closure), &code);
  void* args_dbl[3];
  ffi_type* cls_struct_fields[3];
  ffi_type* cls_struct_fields1[3];
  ffi_type cls_struct_type, cls_struct_type1;
  ffi_type* dbl_arg_types[3];

  float e_dbl = 12.125f;
  struct B f_dbl = { 24.75f, { 31.625f, 32.25f } };

  struct B res_dbl;

  cls_struct_type.size = 0;
  cls_struct_type.alignment = 0;
  cls_struct_type.type = FFI_TYPE_STRUCT;
  cls_struct_type.elements = cls_struct_fields;

  cls_struct_type1.size = 0;
  cls_struct_type1.alignment = 0;
  cls_struct_type1.type = FFI_TYPE_STRUCT;
  cls_struct_type1.elements = cls_struct_fields1;

  cls_struct_fields[0] = &ffi_type_float;
  cls_struct_fields[1] = &ffi_type_float;
  cls_struct_fields[2] = NULL;

  cls_struct_fields1[0] = &ffi_type_float;
  cls_struct_fields1[1] = &cls_struct_type;
  cls_struct_fields1[2] = NULL;


  dbl_arg_types[0] = &ffi_type_float;
  dbl_arg_types[1] = &cls_struct_type1;
  dbl_arg_types[2] = NULL;

  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 2, &cls_struct_type1,
                    dbl_arg_types) == FFI_OK);

  args_dbl[0] = &e_dbl;
  args_dbl[1] = &f_dbl;
  args_dbl[2] = NULL;

  ffi_call(&cif, FFI_FN(B_fn), &res_dbl, args_dbl);
  /* { dg-output "12.125 24.75 31.625 32.25: 36.875 43.75 44.375" } */
  CHECK( res_dbl.x == (e_dbl + f_dbl.x));
  CHECK( res_dbl.y.a == (e_dbl + f_dbl.y.a));
  CHECK( res_dbl.y.b == (e_dbl + f_dbl.y.b));

  CHECK(ffi_prep_closure_loc(pcl, &cif, B_gn, NULL, code) == FFI_OK);

  res_dbl = ((B(*)(float, B))(code))(e_dbl, f_dbl);
  /* { dg-output "\n12.125 24.75 31.625 32.25: 36.875 43.75 44.375" } */
  CHECK( res_dbl.x == (e_dbl + f_dbl.x));
  CHECK( res_dbl.y.a == (e_dbl + f_dbl.y.a));
  CHECK( res_dbl.y.b == (e_dbl + f_dbl.y.b));

  exit(0);
}

