/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 31, 2024.
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
#error "Never use <serializeintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __SERIALIZEINTRIN_H
#define __SERIALIZEINTRIN_H

/// Serialize instruction fetch and execution.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> SERIALIZE </c> instruction.
///
static __inline__ void
__attribute__((__always_inline__, __nodebug__, __target__("serialize")))
_serialize (void)
{
  __builtin_ia32_serialize ();
}

#endif /* __SERIALIZEINTRIN_H */
