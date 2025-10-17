/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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
#if !defined X86GPRINTRIN_H_
#error "Never use <bmiintrin.h> directly; include <x86gprintrin.h> instead."
#endif

#ifndef BMIINTRIN_H_
#define BMIINTRIN_H_

extern __inline unsigned short
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __tzcnt_u16(unsigned short __X) {
  return __builtin_ctz(__X);
}

extern __inline unsigned int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __andn_u32(unsigned int __X, unsigned int __Y) {
  return (~__X & __Y);
}

extern __inline unsigned int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _bextr_u32(unsigned int __X, unsigned int __P, unsigned int __L) {
  return ((__X << (32 - (__L + __P))) >> (32 - __L));
}

extern __inline unsigned int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __bextr_u32(unsigned int __X, unsigned int __Y) {
  unsigned int __P, __L;
  __P = __Y & 0xFF;
  __L = (__Y >> 8) & 0xFF;
  return (_bextr_u32(__X, __P, __L));
}

extern __inline unsigned int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __blsi_u32(unsigned int __X) {
  return (__X & -__X);
}

extern __inline unsigned int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _blsi_u32(unsigned int __X) {
  return __blsi_u32(__X);
}

extern __inline unsigned int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __blsmsk_u32(unsigned int __X) {
  return (__X ^ (__X - 1));
}

extern __inline unsigned int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _blsmsk_u32(unsigned int __X) {
  return __blsmsk_u32(__X);
}

extern __inline unsigned int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __blsr_u32(unsigned int __X) {
  return (__X & (__X - 1));
}

extern __inline unsigned int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _blsr_u32(unsigned int __X) {
  return __blsr_u32(__X);
}

extern __inline unsigned int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __tzcnt_u32(unsigned int __X) {
  return __builtin_ctz(__X);
}

extern __inline unsigned int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _tzcnt_u32(unsigned int __X) {
  return __builtin_ctz(__X);
}

/* use the 64-bit shift, rotate, and count leading zeros instructions
   for long long.  */
#ifdef __PPC64__
extern __inline unsigned long long
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __andn_u64(unsigned long long __X, unsigned long long __Y) {
  return (~__X & __Y);
}

extern __inline unsigned long long
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _bextr_u64(unsigned long long __X, unsigned int __P, unsigned int __L) {
  return ((__X << (64 - (__L + __P))) >> (64 - __L));
}

extern __inline unsigned long long
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __bextr_u64(unsigned long long __X, unsigned long long __Y) {
  unsigned int __P, __L;
  __P = __Y & 0xFF;
  __L = (__Y & 0xFF00) >> 8;
  return (_bextr_u64(__X, __P, __L));
}

extern __inline unsigned long long
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __blsi_u64(unsigned long long __X) {
  return __X & -__X;
}

extern __inline unsigned long long
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _blsi_u64(unsigned long long __X) {
  return __blsi_u64(__X);
}

extern __inline unsigned long long
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __blsmsk_u64(unsigned long long __X) {
  return (__X ^ (__X - 1));
}

extern __inline unsigned long long
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _blsmsk_u64(unsigned long long __X) {
  return __blsmsk_u64(__X);
}

extern __inline unsigned long long
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __blsr_u64(unsigned long long __X) {
  return (__X & (__X - 1));
}

extern __inline unsigned long long
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _blsr_u64(unsigned long long __X) {
  return __blsr_u64(__X);
}

extern __inline unsigned long long
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __tzcnt_u64(unsigned long long __X) {
  return __builtin_ctzll(__X);
}

extern __inline unsigned long long
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _tzcnt_u64(unsigned long long __X) {
  return __builtin_ctzll(__X);
}
#endif /* __PPC64__  */

#endif /* BMIINTRIN_H_ */
