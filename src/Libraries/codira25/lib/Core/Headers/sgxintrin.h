/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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
#if !defined __X86INTRIN_H && !defined __IMMINTRIN_H
#error "Never use <sgxintrin.h> directly; include <x86intrin.h> instead."
#endif

#ifndef __SGXINTRIN_H
#define __SGXINTRIN_H

#if __has_extension(gnu_asm)

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS \
  __attribute__((__always_inline__, __nodebug__,  __target__("sgx")))

static __inline unsigned int __DEFAULT_FN_ATTRS
_enclu_u32(unsigned int __leaf, __SIZE_TYPE__ __d[])
{
  unsigned int __result;
  __asm__ ("enclu"
           : "=a" (__result), "=b" (__d[0]), "=c" (__d[1]), "=d" (__d[2])
           : "a" (__leaf), "b" (__d[0]), "c" (__d[1]), "d" (__d[2])
           : "cc");
  return __result;
}

static __inline unsigned int __DEFAULT_FN_ATTRS
_encls_u32(unsigned int __leaf, __SIZE_TYPE__ __d[])
{
  unsigned int __result;
  __asm__ ("encls"
           : "=a" (__result), "=b" (__d[0]), "=c" (__d[1]), "=d" (__d[2])
           : "a" (__leaf), "b" (__d[0]), "c" (__d[1]), "d" (__d[2])
           : "cc");
  return __result;
}

static __inline unsigned int __DEFAULT_FN_ATTRS
_enclv_u32(unsigned int __leaf, __SIZE_TYPE__ __d[])
{
  unsigned int __result;
  __asm__ ("enclv"
           : "=a" (__result), "=b" (__d[0]), "=c" (__d[1]), "=d" (__d[2])
           : "a" (__leaf), "b" (__d[0]), "c" (__d[1]), "d" (__d[2])
           : "cc");
  return __result;
}

#undef __DEFAULT_FN_ATTRS

#endif /* __has_extension(gnu_asm) */

#endif
