/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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
#include "vpx_config.h"
#include "vp8_rtcd.h"
#include "vpx/vpx_integer.h"

#if HAVE_DSPR2
inline void prefetch_load_int(unsigned char *src) {
  __asm__ __volatile__("pref   0,  0(%[src])   \n\t" : : [src] "r"(src));
}

__inline void vp8_copy_mem16x16_dspr2(unsigned char *RESTRICT src,
                                      int src_stride,
                                      unsigned char *RESTRICT dst,
                                      int dst_stride) {
  int r;
  unsigned int a0, a1, a2, a3;

  for (r = 16; r--;) {
    /* load src data in cache memory */
    prefetch_load_int(src + src_stride);

    /* use unaligned memory load and store */
    __asm__ __volatile__(
        "ulw    %[a0], 0(%[src])            \n\t"
        "ulw    %[a1], 4(%[src])            \n\t"
        "ulw    %[a2], 8(%[src])            \n\t"
        "ulw    %[a3], 12(%[src])           \n\t"
        "sw     %[a0], 0(%[dst])            \n\t"
        "sw     %[a1], 4(%[dst])            \n\t"
        "sw     %[a2], 8(%[dst])            \n\t"
        "sw     %[a3], 12(%[dst])           \n\t"
        : [a0] "=&r"(a0), [a1] "=&r"(a1), [a2] "=&r"(a2), [a3] "=&r"(a3)
        : [src] "r"(src), [dst] "r"(dst));

    src += src_stride;
    dst += dst_stride;
  }
}

__inline void vp8_copy_mem8x8_dspr2(unsigned char *RESTRICT src, int src_stride,
                                    unsigned char *RESTRICT dst,
                                    int dst_stride) {
  int r;
  unsigned int a0, a1;

  /* load src data in cache memory */
  prefetch_load_int(src + src_stride);

  for (r = 8; r--;) {
    /* use unaligned memory load and store */
    __asm__ __volatile__(
        "ulw    %[a0], 0(%[src])            \n\t"
        "ulw    %[a1], 4(%[src])            \n\t"
        "sw     %[a0], 0(%[dst])            \n\t"
        "sw     %[a1], 4(%[dst])            \n\t"
        : [a0] "=&r"(a0), [a1] "=&r"(a1)
        : [src] "r"(src), [dst] "r"(dst));

    src += src_stride;
    dst += dst_stride;
  }
}

__inline void vp8_copy_mem8x4_dspr2(unsigned char *RESTRICT src, int src_stride,
                                    unsigned char *RESTRICT dst,
                                    int dst_stride) {
  int r;
  unsigned int a0, a1;

  /* load src data in cache memory */
  prefetch_load_int(src + src_stride);

  for (r = 4; r--;) {
    /* use unaligned memory load and store */
    __asm__ __volatile__(
        "ulw    %[a0], 0(%[src])            \n\t"
        "ulw    %[a1], 4(%[src])            \n\t"
        "sw     %[a0], 0(%[dst])            \n\t"
        "sw     %[a1], 4(%[dst])            \n\t"
        : [a0] "=&r"(a0), [a1] "=&r"(a1)
        : [src] "r"(src), [dst] "r"(dst));

    src += src_stride;
    dst += dst_stride;
  }
}

#endif
