/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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
#ifndef VPX_VPX_DSP_MIPS_COMMON_DSPR2_H_
#define VPX_VPX_DSP_MIPS_COMMON_DSPR2_H_

#include <assert.h>
#include "./vpx_config.h"
#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif
#if HAVE_DSPR2
#define CROP_WIDTH 512

extern uint8_t *vpx_ff_cropTbl;  // From "vpx_dsp/mips/intrapred4_dspr2.c"

static INLINE void prefetch_load(const unsigned char *src) {
  __asm__ __volatile__("pref   0,  0(%[src])   \n\t" : : [src] "r"(src));
}

/* prefetch data for store */
static INLINE void prefetch_store(unsigned char *dst) {
  __asm__ __volatile__("pref   1,  0(%[dst])   \n\t" : : [dst] "r"(dst));
}

static INLINE void prefetch_load_streamed(const unsigned char *src) {
  __asm__ __volatile__("pref   4,  0(%[src])   \n\t" : : [src] "r"(src));
}

/* prefetch data for store */
static INLINE void prefetch_store_streamed(unsigned char *dst) {
  __asm__ __volatile__("pref   5,  0(%[dst])   \n\t" : : [dst] "r"(dst));
}
#endif  // #if HAVE_DSPR2
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_DSP_MIPS_COMMON_DSPR2_H_
