/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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
/* { dg-output "" { xfail avr32*-*-* } } */
#include "ffitest.h"

struct small_tag
{
  unsigned char a;
  unsigned char b;
};

struct large_tag
{
  unsigned a;
  unsigned b;
  unsigned c;
  unsigned d;
  unsigned e;
};

static void
test_fn (ffi_cif* cif __UNUSED__, void* resp,
	 void** args, void* userdata __UNUSED__)
{
  int n = *(int*)args[0];
  struct small_tag s1 = * (struct small_tag *) args[1];
  struct large_tag l1 = * (struct large_tag *) args[2];
  struct small_tag s2 = * (struct small_tag *) args[3];

  printf ("%d %d %d %d %d %d %d %d %d %d\n", n, s1.a, s1.b,
	  l1.a, l1.b, l1.c, l1.d, l1.e,
	  s2.a, s2.b);
  * (ffi_arg*) resp = 42;
}

int
main (void)
{
  ffi_cif cif;
  void *code;
  ffi_closure *pcl = ffi_closure_alloc (sizeof (ffi_closure), &code);
  ffi_type* arg_types[5];

  ffi_arg res = 0;

  ffi_type s_type;
  ffi_type *s_type_elements[3];

  ffi_type l_type;
  ffi_type *l_type_elements[6];

  struct small_tag s1;
  struct small_tag s2;
  struct large_tag l1;

  int si;

  s_type.size = 0;
  s_type.alignment = 0;
  s_type.type = FFI_TYPE_STRUCT;
  s_type.elements = s_type_elements;

  s_type_elements[0] = &ffi_type_uchar;
  s_type_elements[1] = &ffi_type_uchar;
  s_type_elements[2] = NULL;

  l_type.size = 0;
  l_type.alignment = 0;
  l_type.type = FFI_TYPE_STRUCT;
  l_type.elements = l_type_elements;

  l_type_elements[0] = &ffi_type_uint;
  l_type_elements[1] = &ffi_type_uint;
  l_type_elements[2] = &ffi_type_uint;
  l_type_elements[3] = &ffi_type_uint;
  l_type_elements[4] = &ffi_type_uint;
  l_type_elements[5] = NULL;

  arg_types[0] = &ffi_type_sint;
  arg_types[1] = &s_type;
  arg_types[2] = &l_type;
  arg_types[3] = &s_type;
  arg_types[4] = NULL;

  CHECK(ffi_prep_cif_var(&cif, FFI_DEFAULT_ABI, 1, 4, &ffi_type_sint,
			 arg_types) == FFI_OK);

  si = 4;
  s1.a = 5;
  s1.b = 6;

  s2.a = 20;
  s2.b = 21;

  l1.a = 10;
  l1.b = 11;
  l1.c = 12;
  l1.d = 13;
  l1.e = 14;

  CHECK(ffi_prep_closure_loc(pcl, &cif, test_fn, NULL, code) == FFI_OK);

  res = ((int (*)(int, ...))(code))(si, s1, l1, s2);
  /* { dg-output "4 5 6 10 11 12 13 14 20 21" } */
  printf("res: %d\n", (int) res);
  /* { dg-output "\nres: 42" } */

  exit(0);
}

