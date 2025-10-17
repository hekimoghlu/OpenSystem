/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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
#error "Never use <ptwriteintrin.h> directly; include <x86intrin.h> instead."
#endif

#ifndef __PTWRITEINTRIN_H
#define __PTWRITEINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS \
  __attribute__((__always_inline__, __nodebug__,  __target__("ptwrite")))

static __inline__ void __DEFAULT_FN_ATTRS
_ptwrite32(unsigned int __value) {
  __builtin_ia32_ptwrite32(__value);
}

#ifdef __x86_64__

static __inline__ void __DEFAULT_FN_ATTRS
_ptwrite64(unsigned long long __value) {
  __builtin_ia32_ptwrite64(__value);
}

#endif /* __x86_64__ */

#undef __DEFAULT_FN_ATTRS

#endif /* __PTWRITEINTRIN_H */
