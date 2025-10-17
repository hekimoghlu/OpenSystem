/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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
#ifndef __X86INTRIN_H
#error "Never use <tbmintrin.h> directly; include <x86intrin.h> instead."
#endif

#ifndef __TBMINTRIN_H
#define __TBMINTRIN_H

/* Define the default attributes for the functions in this file. */
#if defined(__cplusplus) && (__cplusplus >= 201103L)
#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__, __target__("tbm"))) constexpr
#else
#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__, __target__("tbm")))
#endif

#define __bextri_u32(a, b) \
  ((unsigned int)__builtin_ia32_bextri_u32((unsigned int)(a), \
                                           (unsigned int)(b)))

static __inline__ unsigned int __DEFAULT_FN_ATTRS
__blcfill_u32(unsigned int __a) {
  return __a & (__a + 1);
}

static __inline__ unsigned int __DEFAULT_FN_ATTRS
__blci_u32(unsigned int __a) {
  return __a | ~(__a + 1);
}

static __inline__ unsigned int __DEFAULT_FN_ATTRS
__blcic_u32(unsigned int __a) {
  return ~__a & (__a + 1);
}

static __inline__ unsigned int __DEFAULT_FN_ATTRS
__blcmsk_u32(unsigned int __a) {
  return __a ^ (__a + 1);
}

static __inline__ unsigned int __DEFAULT_FN_ATTRS
__blcs_u32(unsigned int __a) {
  return __a | (__a + 1);
}

static __inline__ unsigned int __DEFAULT_FN_ATTRS
__blsfill_u32(unsigned int __a) {
  return __a | (__a - 1);
}

static __inline__ unsigned int __DEFAULT_FN_ATTRS
__blsic_u32(unsigned int __a) {
  return ~__a | (__a - 1);
}

static __inline__ unsigned int __DEFAULT_FN_ATTRS
__t1mskc_u32(unsigned int __a) {
  return ~__a | (__a + 1);
}

static __inline__ unsigned int __DEFAULT_FN_ATTRS
__tzmsk_u32(unsigned int __a) {
  return ~__a & (__a - 1);
}

#ifdef __x86_64__
#define __bextri_u64(a, b) \
  ((unsigned long long)__builtin_ia32_bextri_u64((unsigned long long)(a), \
                                                 (unsigned long long)(b)))

static __inline__ unsigned long long __DEFAULT_FN_ATTRS
__blcfill_u64(unsigned long long __a) {
  return __a & (__a + 1);
}

static __inline__ unsigned long long __DEFAULT_FN_ATTRS
__blci_u64(unsigned long long __a) {
  return __a | ~(__a + 1);
}

static __inline__ unsigned long long __DEFAULT_FN_ATTRS
__blcic_u64(unsigned long long __a) {
  return ~__a & (__a + 1);
}

static __inline__ unsigned long long __DEFAULT_FN_ATTRS
__blcmsk_u64(unsigned long long __a) {
  return __a ^ (__a + 1);
}

static __inline__ unsigned long long __DEFAULT_FN_ATTRS
__blcs_u64(unsigned long long __a) {
  return __a | (__a + 1);
}

static __inline__ unsigned long long __DEFAULT_FN_ATTRS
__blsfill_u64(unsigned long long __a) {
  return __a | (__a - 1);
}

static __inline__ unsigned long long __DEFAULT_FN_ATTRS
__blsic_u64(unsigned long long __a) {
  return ~__a | (__a - 1);
}

static __inline__ unsigned long long __DEFAULT_FN_ATTRS
__t1mskc_u64(unsigned long long __a) {
  return ~__a | (__a + 1);
}

static __inline__ unsigned long long __DEFAULT_FN_ATTRS
__tzmsk_u64(unsigned long long __a) {
  return ~__a & (__a - 1);
}
#endif

#undef __DEFAULT_FN_ATTRS

#endif /* __TBMINTRIN_H */
