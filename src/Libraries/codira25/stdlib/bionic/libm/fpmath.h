/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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
// ANDROID changed:
// - keep only little endian variants as they're the only one supported.
// - add long double structures here instead of _fpmath.h.
// - android uses 128 bits long doubles for LP64, so the structure and macros
//   were reworked for the quad precision ieee representation.

#pragma once

#include <endian.h>

union IEEEf2bits {
  float f;
  struct {
    unsigned int man   :23;
    unsigned int exp   :8;
    unsigned int sign  :1;
  } bits;
};

#define DBL_MANH_SIZE  20
#define DBL_MANL_SIZE  32

union IEEEd2bits {
  double  d;
  struct {
    unsigned int manl  :32;
    unsigned int manh  :20;
    unsigned int exp   :11;
    unsigned int sign  :1;
  } bits;
};

#ifdef __LP64__

union IEEEl2bits {
  long double e;
  struct {
    unsigned long manl  :64;
    unsigned long manh  :48;
    unsigned int  exp   :15;
    unsigned int  sign  :1;
  } bits;
  struct {
    unsigned long manl     :64;
    unsigned long manh     :48;
    unsigned int  expsign  :16;
  } xbits;
};

#define LDBL_NBIT  0
#define LDBL_IMPLICIT_NBIT
#define mask_nbit_l(u)  ((void)0)

#define LDBL_MANH_SIZE  48
#define LDBL_MANL_SIZE  64

#define LDBL_TO_ARRAY32(u, a) do {           \
  (a)[0] = (uint32_t)(u).bits.manl;          \
  (a)[1] = (uint32_t)((u).bits.manl >> 32);  \
  (a)[2] = (uint32_t)(u).bits.manh;          \
  (a)[3] = (uint32_t)((u).bits.manh >> 32);  \
} while(0)

#endif // __LP64__
