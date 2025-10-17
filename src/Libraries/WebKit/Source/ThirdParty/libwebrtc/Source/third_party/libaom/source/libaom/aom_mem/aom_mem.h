/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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
#ifndef AOM_AOM_MEM_AOM_MEM_H_
#define AOM_AOM_MEM_AOM_MEM_H_

#include "aom/aom_integer.h"
#include "config/aom_config.h"

#if defined(__uClinux__)
#include <lddk.h>
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#ifndef AOM_MAX_ALLOCABLE_MEMORY
#if SIZE_MAX > (1ULL << 32)
#define AOM_MAX_ALLOCABLE_MEMORY 8589934592  // 8 GB
#else
// For 32-bit targets keep this below INT_MAX to avoid valgrind warnings.
#define AOM_MAX_ALLOCABLE_MEMORY ((1ULL << 31) - (1 << 16))
#endif
#endif

void *aom_memalign(size_t align, size_t size);
void *aom_malloc(size_t size);
void *aom_calloc(size_t num, size_t size);
void aom_free(void *memblk);

static inline void *aom_memset16(void *dest, int val, size_t length) {
  size_t i;
  uint16_t *dest16 = (uint16_t *)dest;
  for (i = 0; i < length; i++) *dest16++ = val;
  return dest;
}

/*returns an addr aligned to the byte boundary specified by align*/
#define aom_align_addr(addr, align) \
  (void *)(((uintptr_t)(addr) + ((align)-1)) & ~(uintptr_t)((align)-1))

#include <string.h>

#ifdef AOM_MEM_PLTFRM
#include AOM_MEM_PLTFRM
#endif

#if CONFIG_DEBUG
#define AOM_CHECK_MEM_ERROR(error_info, lval, expr)                         \
  do {                                                                      \
    lval = (expr);                                                          \
    if (!lval)                                                              \
      aom_internal_error(error_info, AOM_CODEC_MEM_ERROR,                   \
                         "Failed to allocate " #lval " at %s:%d", __FILE__, \
                         __LINE__);                                         \
  } while (0)
#else
#define AOM_CHECK_MEM_ERROR(error_info, lval, expr)       \
  do {                                                    \
    lval = (expr);                                        \
    if (!lval)                                            \
      aom_internal_error(error_info, AOM_CODEC_MEM_ERROR, \
                         "Failed to allocate " #lval);    \
  } while (0)
#endif

#if defined(__cplusplus)
}
#endif

#endif  // AOM_AOM_MEM_AOM_MEM_H_
