/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 22, 2023.
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
#pragma once

#include <sys/cdefs.h>

#include <stdint.h>
#include <sys/types.h>

#if !defined(__BIONIC_SWAB_INLINE)
#define __BIONIC_SWAB_INLINE static __inline
#endif

__BEGIN_DECLS

__BIONIC_SWAB_INLINE void swab(const void* _Nonnull __void_src, void* _Nonnull __void_dst, ssize_t __byte_count) {
  const uint8_t* __src = __BIONIC_CAST(static_cast, const uint8_t*, __void_src);
  uint8_t* __dst = __BIONIC_CAST(static_cast, uint8_t*, __void_dst);
  while (__byte_count > 1) {
    uint8_t x = *__src++;
    uint8_t y = *__src++;
    *__dst++ = y;
    *__dst++ = x;
    __byte_count -= 2;
  }
}

__END_DECLS
