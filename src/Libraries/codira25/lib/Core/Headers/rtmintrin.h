/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 31, 2022.
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
#error "Never use <rtmintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __RTMINTRIN_H
#define __RTMINTRIN_H

#define _XBEGIN_STARTED   (~0u)
#define _XABORT_EXPLICIT  (1 << 0)
#define _XABORT_RETRY     (1 << 1)
#define _XABORT_CONFLICT  (1 << 2)
#define _XABORT_CAPACITY  (1 << 3)
#define _XABORT_DEBUG     (1 << 4)
#define _XABORT_NESTED    (1 << 5)
#define _XABORT_CODE(x)   (((x) >> 24) & 0xFF)

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__, __target__("rtm")))

static __inline__ unsigned int __DEFAULT_FN_ATTRS
_xbegin(void)
{
  return (unsigned int)__builtin_ia32_xbegin();
}

static __inline__ void __DEFAULT_FN_ATTRS
_xend(void)
{
  __builtin_ia32_xend();
}

#define _xabort(imm) __builtin_ia32_xabort((imm))

#undef __DEFAULT_FN_ATTRS

#endif /* __RTMINTRIN_H */
