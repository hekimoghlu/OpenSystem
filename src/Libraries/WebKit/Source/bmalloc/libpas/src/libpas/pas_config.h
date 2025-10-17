/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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
#ifndef PAS_CONFIG_H
#define PAS_CONFIG_H

#include "pas_config_prefix.h"

#include "stdbool.h"

#define PAS_LOG_NONE (0)
#define PAS_LOG_HEAP_INFRASTRUCTURE (1 << 0)
#define PAS_LOG_BOOTSTRAP_HEAPS (1 << 1)
#define PAS_LOG_IMMORTAL_HEAPS (1 << 2)
#define PAS_LOG_SEGREGATED_HEAPS (1 << 3)
#define PAS_LOG_BITFIT_HEAPS (1 << 4)
#define PAS_LOG_LARGE_HEAPS (1 << 5)
#define PAS_LOG_JIT_HEAPS (1 << 6)
#define PAS_LOG_OTHER (1 << 7)  /* Heap-type agnostic */

#define PAS_LOG_LEVEL (PAS_LOG_NONE)
#define PAS_SHOULD_LOG(level) (PAS_LOG_LEVEL & level)

#define LIBPAS_ENABLED 1

#if defined(PAS_BMALLOC)
#include "BPlatform.h"
#if !BENABLE(LIBPAS)
#undef LIBPAS_ENABLED
#define LIBPAS_ENABLED 0
#endif
#endif

#if ((PAS_OS(DARWIN) && __PAS_ARM64 && !__PAS_ARM64E) || PAS_PLATFORM(PLAYSTATION)) && defined(NDEBUG)
#define PAS_ENABLE_ASSERT 0
#else
#define PAS_ENABLE_ASSERT 1
#endif
#define PAS_ENABLE_TESTING __PAS_ENABLE_TESTING

#define PAS_ARM64 __PAS_ARM64
#define PAS_ARM32 __PAS_ARM32

#define PAS_ARM __PAS_ARM

#define PAS_RISCV __PAS_RISCV

#define PAS_ADDRESS_BITS                 48

#if PAS_ARM || PAS_PLATFORM(PLAYSTATION)
#define PAS_MAX_GRANULES                 256
#else
#define PAS_MAX_GRANULES                 1024
#endif

#define PAS_INTERNAL_MIN_ALIGN_SHIFT     3
#define PAS_INTERNAL_MIN_ALIGN           ((size_t)1 << PAS_INTERNAL_MIN_ALIGN_SHIFT)

#if defined(PAS_BMALLOC)
#define PAS_ENABLE_THINGY                0
#define PAS_ENABLE_ISO                   0
#define PAS_ENABLE_ISO_TEST              0
#define PAS_ENABLE_MINALIGN32            0
#define PAS_ENABLE_PAGESIZE64K           0
#define PAS_ENABLE_BMALLOC               1
#define PAS_ENABLE_HOTBIT                0
#define PAS_ENABLE_JIT                   1
#elif defined(PAS_LIBMALLOC)
#define PAS_ENABLE_THINGY                0
#define PAS_ENABLE_ISO                   1
#define PAS_ENABLE_ISO_TEST              0
#define PAS_ENABLE_MINALIGN32            0
#define PAS_ENABLE_PAGESIZE64K           0
#define PAS_ENABLE_BMALLOC               0
#define PAS_ENABLE_HOTBIT                0
#define PAS_ENABLE_JIT                   0
#else /* PAS_LIBMALLOC -> so !defined(PAS_BMALLOC) && !defined(PAS_LIBMALLOC) */
#define PAS_ENABLE_THINGY                1
#define PAS_ENABLE_ISO                   1
#define PAS_ENABLE_ISO_TEST              1
#define PAS_ENABLE_MINALIGN32            1
#define PAS_ENABLE_PAGESIZE64K           1
#define PAS_ENABLE_BMALLOC               1
#define PAS_ENABLE_HOTBIT                1
#define PAS_ENABLE_JIT                   1
#endif /* PAS_LIBMALLOC -> so end of !defined(PAS_BMALLOC) && !defined(PAS_LIBMALLOC) */

#define PAS_COMPACT_PTR_SIZE             3
#define PAS_COMPACT_PTR_BITS             (PAS_COMPACT_PTR_SIZE << 3)
#define PAS_COMPACT_PTR_MASK             ((uintptr_t)(((uint64_t)1 \
                                                       << (PAS_COMPACT_PTR_BITS & 63)) - 1))

#define PAS_ALLOCATOR_INDEX_BYTES        4

#if PAS_OS(DARWIN) || PAS_PLATFORM(PLAYSTATION)
#define PAS_USE_SPINLOCKS                0
#else
#define PAS_USE_SPINLOCKS                1
#endif

#endif /* PAS_CONFIG_H */

