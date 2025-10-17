/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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
#error "Never use <enqcmdintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __ENQCMDINTRIN_H
#define __ENQCMDINTRIN_H

/* Define the default attributes for the functions in this file */
#define _DEFAULT_FN_ATTRS \
  __attribute__((__always_inline__, __nodebug__, __target__("enqcmd")))

/// Reads 64-byte command pointed by \a __src, formats 64-byte enqueue store
///    data, and performs 64-byte enqueue store to memory pointed by \a __dst.
///    This intrinsics may only be used in User mode.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsics corresponds to the <c> ENQCMD </c> instruction.
///
/// \param __dst
///    Pointer to the destination of the enqueue store.
/// \param __src
///    Pointer to 64-byte command data.
/// \returns If the command data is successfully written to \a __dst then 0 is
///    returned. Otherwise 1 is returned.
static __inline__ int _DEFAULT_FN_ATTRS
_enqcmd (void *__dst, const void *__src)
{
  return __builtin_ia32_enqcmd(__dst, __src);
}

/// Reads 64-byte command pointed by \a __src, formats 64-byte enqueue store
///    data, and performs 64-byte enqueue store to memory pointed by \a __dst
///    This intrinsic may only be used in Privileged mode.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsics corresponds to the <c> ENQCMDS </c> instruction.
///
/// \param __dst
///    Pointer to the destination of the enqueue store.
/// \param __src
///    Pointer to 64-byte command data.
/// \returns If the command data is successfully written to \a __dst then 0 is
///    returned. Otherwise 1 is returned.
static __inline__ int _DEFAULT_FN_ATTRS
_enqcmds (void *__dst, const void *__src)
{
  return __builtin_ia32_enqcmds(__dst, __src);
}

#undef _DEFAULT_FN_ATTRS

#endif /* __ENQCMDINTRIN_H */
