/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 1, 2023.
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
#error "Never use <xsavesintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __XSAVESINTRIN_H
#define __XSAVESINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__,  __target__("xsaves")))

static __inline__ void __DEFAULT_FN_ATTRS
_xsaves(void *__p, unsigned long long __m) {
  __builtin_ia32_xsaves(__p, __m);
}

static __inline__ void __DEFAULT_FN_ATTRS
_xrstors(void *__p, unsigned long long __m) {
  __builtin_ia32_xrstors(__p, __m);
}

#ifdef __x86_64__
static __inline__ void __DEFAULT_FN_ATTRS
_xrstors64(void *__p, unsigned long long __m) {
  __builtin_ia32_xrstors64(__p, __m);
}

static __inline__ void __DEFAULT_FN_ATTRS
_xsaves64(void *__p, unsigned long long __m) {
  __builtin_ia32_xsaves64(__p, __m);
}
#endif

#undef __DEFAULT_FN_ATTRS

#endif
