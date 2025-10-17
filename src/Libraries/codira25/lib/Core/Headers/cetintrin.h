/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
#ifndef __IMMINTRIN_H
#error "Never use <cetintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __CETINTRIN_H
#define __CETINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__, __target__("shstk")))

static __inline__ void __DEFAULT_FN_ATTRS _incsspd(int __a) {
  __builtin_ia32_incsspd((unsigned int)__a);
}

#ifdef __x86_64__
static __inline__ void __DEFAULT_FN_ATTRS _incsspq(unsigned long long __a) {
  __builtin_ia32_incsspq(__a);
}
#endif /* __x86_64__ */

#ifdef __x86_64__
static __inline__ void __DEFAULT_FN_ATTRS _inc_ssp(unsigned int __a) {
  __builtin_ia32_incsspq(__a);
}
#else /* __x86_64__ */
static __inline__ void __DEFAULT_FN_ATTRS _inc_ssp(unsigned int __a) {
  __builtin_ia32_incsspd(__a);
}
#endif /* __x86_64__ */

static __inline__ unsigned int __DEFAULT_FN_ATTRS _rdsspd(unsigned int __a) {
  return __builtin_ia32_rdsspd(__a);
}

static __inline__ unsigned int __DEFAULT_FN_ATTRS _rdsspd_i32(void) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
  unsigned int t;
  return __builtin_ia32_rdsspd(t);
#pragma clang diagnostic pop
}

#ifdef __x86_64__
static __inline__ unsigned long long __DEFAULT_FN_ATTRS _rdsspq(unsigned long long __a) {
  return __builtin_ia32_rdsspq(__a);
}

static __inline__ unsigned long long __DEFAULT_FN_ATTRS _rdsspq_i64(void) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
  unsigned long long t;
  return __builtin_ia32_rdsspq(t);
#pragma clang diagnostic pop
}
#endif /* __x86_64__ */

#ifdef __x86_64__
static __inline__ unsigned long long __DEFAULT_FN_ATTRS _get_ssp(void) {
  return __builtin_ia32_rdsspq(0);
}
#else /* __x86_64__ */
static __inline__ unsigned int __DEFAULT_FN_ATTRS _get_ssp(void) {
  return __builtin_ia32_rdsspd(0);
}
#endif /* __x86_64__ */

static __inline__ void __DEFAULT_FN_ATTRS _saveprevssp(void) {
  __builtin_ia32_saveprevssp();
}

static __inline__ void __DEFAULT_FN_ATTRS _rstorssp(void * __p) {
  __builtin_ia32_rstorssp(__p);
}

static __inline__ void __DEFAULT_FN_ATTRS _wrssd(unsigned int __a, void * __p) {
  __builtin_ia32_wrssd(__a, __p);
}

#ifdef __x86_64__
static __inline__ void __DEFAULT_FN_ATTRS _wrssq(unsigned long long __a, void * __p) {
  __builtin_ia32_wrssq(__a, __p);
}
#endif /* __x86_64__ */

static __inline__ void __DEFAULT_FN_ATTRS _wrussd(unsigned int __a, void * __p) {
  __builtin_ia32_wrussd(__a, __p);
}

#ifdef __x86_64__
static __inline__ void __DEFAULT_FN_ATTRS _wrussq(unsigned long long __a, void * __p) {
  __builtin_ia32_wrussq(__a, __p);
}
#endif /* __x86_64__ */

static __inline__ void __DEFAULT_FN_ATTRS _setssbsy(void) {
  __builtin_ia32_setssbsy();
}

static __inline__ void __DEFAULT_FN_ATTRS _clrssbsy(void * __p) {
  __builtin_ia32_clrssbsy(__p);
}

#undef __DEFAULT_FN_ATTRS

#endif /* __CETINTRIN_H */
