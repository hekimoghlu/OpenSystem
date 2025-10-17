/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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

typedef struct my_ffi_struct {
  double a;
  double b;
  double c;
} my_ffi_struct;

my_ffi_struct callee(struct my_ffi_struct a1, struct my_ffi_struct a2)
{
  struct my_ffi_struct result;
  result.a = a1.a + a2.a;
  result.b = a1.b + a2.b;
  result.c = a1.c + a2.c;


  printf("%g %g %g %g %g %g: %g %g %g\n", a1.a, a1.b, a1.c,
	 a2.a, a2.b, a2.c, result.a, result.b, result.c);

  return result;
}

void stub(ffi_cif* cif, void* resp, void** args, void* userdata)
{
  struct my_ffi_struct a1;
  struct my_ffi_struct a2;

  a1 = *(struct my_ffi_struct*)(args[0]);
  a2 = *(struct my_ffi_struct*)(args[1]);

  *(my_ffi_struct *)resp = callee(a1, a2);
}


int main(void)
{
  ffi_type* my_ffi_struct_fields[4];
  ffi_type my_ffi_struct_type;
  ffi_cif cif;
#ifndef USING_MMAP
  static ffi_closure cl;
#endif
  ffi_closure *pcl;
  void* args[4];
  ffi_type* arg_types[3];

#ifdef USING_MMAP
  pcl = allocate_mmap (sizeof(ffi_closure));
#else
  pcl = &cl;
#endif

  struct my_ffi_struct g = { 1.0, 2.0, 3.0 };
  struct my_ffi_struct f = { 1.0, 2.0, 3.0 };
  struct my_ffi_struct res;

  my_ffi_struct_type.size = 0;
  my_ffi_struct_type.alignment = 0;
  my_ffi_struct_type.type = FFI_TYPE_STRUCT;
  my_ffi_struct_type.elements = my_ffi_struct_fields;

  my_ffi_struct_fields[0] = &ffi_type_double;
  my_ffi_struct_fields[1] = &ffi_type_double;
  my_ffi_struct_fields[2] = &ffi_type_double;
  my_ffi_struct_fields[3] = NULL;

  arg_types[0] = &my_ffi_struct_type;
  arg_types[1] = &my_ffi_struct_type;
  arg_types[2] = NULL;

  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 2, &my_ffi_struct_type,
		     arg_types) == FFI_OK);

  args[0] = &g;
  args[1] = &f;
  args[2] = NULL;
  ffi_call(&cif, FFI_FN(callee), &res, args);
  /* { dg-output "1 2 3 1 2 3: 2 4 6" } */
  printf("res: %g %g %g\n", res.a, res.b, res.c);
  /* { dg-output "\nres: 2 4 6" } */

  CHECK(ffi_prep_closure(pcl, &cif, stub, NULL) == FFI_OK);

  res = ((my_ffi_struct(*)(struct my_ffi_struct, struct my_ffi_struct))(pcl))(g, f);
  /* { dg-output "\n1 2 3 1 2 3: 2 4 6" } */
  printf("res: %g %g %g\n", res.a, res.b, res.c);
  /* { dg-output "\nres: 2 4 6" } */

  exit(0);;
}
