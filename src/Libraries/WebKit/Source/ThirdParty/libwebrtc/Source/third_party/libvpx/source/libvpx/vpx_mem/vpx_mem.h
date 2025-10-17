/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
#ifndef VPX_VPX_MEM_VPX_MEM_H_
#define VPX_VPX_MEM_VPX_MEM_H_

#include "vpx_config.h"
#if defined(__uClinux__)
#include <lddk.h>
#endif

#include <stdlib.h>
#include <stddef.h>

#include "vpx/vpx_integer.h"

#if defined(__cplusplus)
extern "C" {
#endif

void *vpx_memalign(size_t align, size_t size);
void *vpx_malloc(size_t size);
void *vpx_calloc(size_t num, size_t size);
void vpx_free(void *memblk);

#if CONFIG_VP9_HIGHBITDEPTH
static INLINE void *vpx_memset16(void *dest, int val, size_t length) {
  size_t i;
  uint16_t *dest16 = (uint16_t *)dest;
  for (i = 0; i < length; i++) *dest16++ = val;
  return dest;
}
#endif

#include <string.h>

#ifdef VPX_MEM_PLTFRM
#include VPX_MEM_PLTFRM
#endif

#if defined(__cplusplus)
}
#endif

#endif  // VPX_VPX_MEM_VPX_MEM_H_
