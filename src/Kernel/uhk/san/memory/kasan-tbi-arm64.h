/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 19, 2023.
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
#include <mach/vm_param.h>
#include "kasan-tbi.h"

#ifndef _KASAN_TBI_ARM64_H_
#define _KASAN_TBI_ARM64_H_

#if KASAN_LIGHT
#define STOLEN_MEM_PERCENT      5UL
#else
#define STOLEN_MEM_PERCENT      7UL
#endif /* KASAN_LIGHT */
/* No need for quarantine with KASAN-TBI */
#define STOLEN_MEM_BYTES            MiB(20)

/* Defined in makedefs/MakeInc.def */
#ifndef KASAN_OFFSET_ARM64
#define KASAN_OFFSET_ARM64      0xf000000000000000ULL
#endif  /* KASAN_OFFSET_ARM64 */

#if defined(ARM_LARGE_MEMORY)
#define KASAN_SHADOW_MIN        (VM_MAX_KERNEL_ADDRESS+1)
#define KASAN_SHADOW_MAX        0xffffffffffffffffULL
#elif defined(KERNEL_INTEGRITY_KTRR) || defined(KERNEL_INTEGRITY_CTRR)
#define KASAN_SHADOW_MIN        0xfffffffdc0000000ULL
#define KASAN_SHADOW_MAX        0xffffffffc0000000ULL
#else /* defined(KERNEL_INTEGRITY_KTRR) || defined(KERNEL_INTEGRITY_CTRR) */
#define KASAN_SHADOW_MIN        0xfffffffe00000000ULL
#define KASAN_SHADOW_MAX        0xffffffffc0000000ULL
#endif

#endif /* _KASAN_TBI_ARM64_H_ */
