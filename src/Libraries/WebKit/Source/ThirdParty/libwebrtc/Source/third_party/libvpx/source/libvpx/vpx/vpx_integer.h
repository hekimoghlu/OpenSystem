/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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
#ifndef VPX_VPX_VPX_INTEGER_H_
#define VPX_VPX_VPX_INTEGER_H_

/* get ptrdiff_t, size_t, wchar_t, NULL */
#include <stddef.h>  // IWYU pragma: export

#if defined(_MSC_VER)
#define VPX_FORCE_INLINE __forceinline
#define VPX_INLINE __inline
#else
#define VPX_FORCE_INLINE __inline__ __attribute__((always_inline))
// TODO(jbb): Allow a way to force inline off for older compilers.
#define VPX_INLINE inline
#endif

/* Assume platforms have the C99 standard integer types. */

#if defined(__cplusplus)
#if !defined(__STDC_FORMAT_MACROS)
#define __STDC_FORMAT_MACROS
#endif
#if !defined(__STDC_LIMIT_MACROS)
#define __STDC_LIMIT_MACROS
#endif
#endif  // __cplusplus

#include <inttypes.h>  // IWYU pragma: export
#include <stdint.h>    // IWYU pragma: export

#endif  // VPX_VPX_VPX_INTEGER_H_
