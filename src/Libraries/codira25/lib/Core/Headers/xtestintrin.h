/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 14, 2024.
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
#error "Never use <xtestintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __XTESTINTRIN_H
#define __XTESTINTRIN_H

/* xtest returns non-zero if the instruction is executed within an RTM or active
 * HLE region. */
/* FIXME: This can be an either or for RTM/HLE. Deal with this when HLE is
 * supported. */
static __inline__ int
    __attribute__((__always_inline__, __nodebug__, __target__("rtm")))
    _xtest(void) {
  return __builtin_ia32_xtest();
}

#endif
