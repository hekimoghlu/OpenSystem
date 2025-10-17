/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 14, 2025.
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

struct s {
  int s32;
  float f32;
  signed char s8;
};

struct s make_s(void) {
  struct s r;
  r.s32 = 0x1234;
  r.f32 = 7.0;
  r.s8  = 0x78;
  return r;
}

int main() {
  ffi_cif cif;
  struct s r;
  ffi_type rtype;
  ffi_type* s_fields[] = {
    &ffi_type_sint,
    &ffi_type_float,
    &ffi_type_schar,
    NULL,
  };

  rtype.size      = 0;
  rtype.alignment = 0,
  rtype.type      = FFI_TYPE_STRUCT,
  rtype.elements  = s_fields,

  r.s32 = 0xbad;
  r.f32 = 999.999;
  r.s8  = 0x51;

  // Here we emulate the following call:
  //r = make_s();

  CHECK(ffi_prep_cif(&cif, FFI_DEFAULT_ABI, 0, &rtype, NULL) == FFI_OK);
  ffi_call(&cif, FFI_FN(make_s), &r, NULL);

  CHECK(r.s32 == 0x1234);
  CHECK(r.f32 == 7.0);
  CHECK(r.s8  == 0x78);
  exit(0);
}

