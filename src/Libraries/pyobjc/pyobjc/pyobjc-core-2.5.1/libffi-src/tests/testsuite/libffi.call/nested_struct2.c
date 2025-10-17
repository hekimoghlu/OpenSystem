/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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

#if LONG_MAX == 2147483647
#define ffi_type_mylong ffi_type_uint32
#else
#if LONG_MAX == 9223372036854775807
#define ffi_type_mylong ffi_type_uint64
#else
#error "Error, size LONG not defined as expected"
#endif
#endif

typedef struct A {
  unsigned long a;
  unsigned char b;
} A;

typedef struct B {
  struct A x;
  unsigned char y;
} B;

B B_fn(struct A b0, struct B b1)
{
  struct B result;

  result.x.a = b0.a + b1.x.a;
  result.x.b = b0.b + b1.x.b + b1.y;
  result.y = b0.b + b1.x.b;

  printf("%d %d %d %d %d: %d %d %d\n", b0.a, b0.b, b1.x.a, b1.x.b, b1.y,
	 result.x.a, result.x.b, result.y);

  return result;
}

static void
B_gn(ffi_cif* cif, void* resp, void** args, void* userdata)
{
  struct A b0;
  struct B b1;

  b0 = *(struct A*)(args[0]);
  b1 = *(struct B*)(args[1]);

  *(B*)resp = B_fn(b0, b1);
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
  ffi_type* cls_struct_fields1[3];
  ffi_type cls_struct_type, cls_struct_type1;
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

  cls_struct_type1.size = 0;
  cls_struct_type1.alignment = 0;
  cls_struct_type1.type = FFI_TYPE_STRUCT;
  cls_struct_type1.elements = cls_struct_fields1;

  struct A e_dbl = { 1, 7};
  struct B f_dbl = {{12 , 127}, 99};

  struct B res_dbl;

  cls_struct_fields[0] = &ffi_type_mylong;
  cls_struct_fields[1] = &ffi_type_uchar;
  cls_struct_fields[2] = NULL;

  cls_struct_fields1[0] = &cls_struct_type;
  cls_struct_fields1[1] = &ffi_type_uchar;
  cls_struct_fields1[2] = NULL;


  dbl_arg_types[0] = &cls_struct_type;
  dbl_arg_types[1] = &cls_struct_type1;
  dbl_arg_types[2] = NULL;

  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 2, &cls_struct_type1,
		     dbl_arg_types) == FFI_OK);

  args_dbl[0] = &e_dbl;
  args_dbl[1] = &f_dbl;
  args_dbl[2] = NULL;

  ffi_call(&cif, FFI_FN(B_fn), &res_dbl, args_dbl);
  /* { dg-output "1 7 12 127 99: 13 233 134" } */
  CHECK( res_dbl.x.a == (e_dbl.a + f_dbl.x.a));
  CHECK( res_dbl.x.b == (e_dbl.b + f_dbl.x.b + f_dbl.y));
  CHECK( res_dbl.y == (e_dbl.b + f_dbl.x.b));


  CHECK(ffi_prep_closure(pcl, &cif, B_gn, NULL) == FFI_OK);

  res_dbl = ((B(*)(A, B))(pcl))(e_dbl, f_dbl);
  /* { dg-output "\n1 7 12 127 99: 13 233 134" } */
  CHECK( res_dbl.x.a == (e_dbl.a + f_dbl.x.a));
  CHECK( res_dbl.x.b == (e_dbl.b + f_dbl.x.b + f_dbl.y));
  CHECK( res_dbl.y == (e_dbl.b + f_dbl.x.b));
  exit(0);
}
