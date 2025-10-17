/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 10, 2025.
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
#error "Never use <amxmovrsintrin.h> directly; include <immintrin.h> instead."
#endif /* __IMMINTRIN_H */

#ifndef __AMXMOVRSINTRIN_H
#define __AMXMOVRSINTRIN_H
#ifdef __x86_64__

#define __DEFAULT_FN_ATTRS_MOVRS                                               \
  __attribute__((__always_inline__, __nodebug__, __target__("amx-movrs")))

#define _tile_loaddrs(dst, base, stride)                                       \
  __builtin_ia32_tileloaddrs64((dst), ((const void *)(base)),                  \
                               (__SIZE_TYPE__)(stride))
#define _tile_stream_loaddrs(dst, base, stride)                                \
  __builtin_ia32_tileloaddrst164((dst), ((const void *)(base)),                \
                                 (__SIZE_TYPE__)(stride))
static __inline__ _tile1024i __DEFAULT_FN_ATTRS_MOVRS
_tile_loaddrs_internal(unsigned short m, unsigned short n, const void *base,
                       __SIZE_TYPE__ stride) {
  return __builtin_ia32_tileloaddrs64_internal(m, n, base,
                                               (__SIZE_TYPE__)(stride));
}
static __inline__ _tile1024i __DEFAULT_FN_ATTRS_MOVRS
_tile_loaddrst1_internal(unsigned short m, unsigned short n, const void *base,
                         __SIZE_TYPE__ stride) {
  return __builtin_ia32_tileloaddrst164_internal(m, n, base,
                                                 (__SIZE_TYPE__)(stride));
}
static __inline__ void __DEFAULT_FN_ATTRS_MOVRS
__tile_loaddrs(__tile1024i *dst, const void *base, __SIZE_TYPE__ stride) {
  dst->tile = _tile_loaddrs_internal(dst->row, dst->col, base, stride);
}
static __inline__ void __DEFAULT_FN_ATTRS_MOVRS __tile_stream_loaddrs(
    __tile1024i *dst, const void *base, __SIZE_TYPE__ stride) {
  dst->tile = _tile_loaddrst1_internal(dst->row, dst->col, base, stride);
}
#undef __DEFAULT_FN_ATTRS_MOVRS
#endif /* __x86_64__ */
#endif /* __AMXMOVRSINTRIN_H */
