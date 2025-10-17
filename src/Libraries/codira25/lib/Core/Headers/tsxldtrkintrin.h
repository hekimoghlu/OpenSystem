/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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
#error "Never use <tsxldtrkintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __TSXLDTRKINTRIN_H
#define __TSXLDTRKINTRIN_H

/* Define the default attributes for the functions in this file */
#define _DEFAULT_FN_ATTRS \
  __attribute__((__always_inline__, __nodebug__, __target__("tsxldtrk")))

/// Marks the start of an TSX (RTM) suspend load address tracking region. If
///    this intrinsic is used inside a transactional region, subsequent loads
///    are not added to the read set of the transaction. If it's used inside a
///    suspend load address tracking region it will cause transaction abort.
///    If it's used outside of a transactional region it behaves like a NOP.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c XSUSLDTRK instruction.
///
static __inline__ void _DEFAULT_FN_ATTRS
_xsusldtrk (void)
{
    __builtin_ia32_xsusldtrk();
}

/// Marks the end of an TSX (RTM) suspend load address tracking region. If this
///    intrinsic is used inside a suspend load address tracking region it will
///    end the suspend region and all following load addresses will be added to
///    the transaction read set. If it's used inside an active transaction but
///    not in a suspend region it will cause transaction abort. If it's used
///    outside of a transactional region it behaves like a NOP.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the \c XRESLDTRK instruction.
///
static __inline__ void _DEFAULT_FN_ATTRS
_xresldtrk (void)
{
    __builtin_ia32_xresldtrk();
}

#undef _DEFAULT_FN_ATTRS

#endif /* __TSXLDTRKINTRIN_H */
