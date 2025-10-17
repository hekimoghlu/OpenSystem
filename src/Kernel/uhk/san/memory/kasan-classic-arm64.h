/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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
#include "kasan-classic.h"

#ifndef _KASAN_CLASSIC_ARM64_H_
#define _KASAN_CLASSIC_ARM64_H_

/*
 * ARM64 configuration for KASAN-CLASSIC.
 */

#define STOLEN_MEM_PERCENT      13UL
/* XXX this is for quarantine, should move to allocator-specific quarantine */
#define STOLEN_MEM_BYTES            MiB(40)

/* Defined in makedefs/MakeInc.def */
#ifndef KASAN_OFFSET_ARM64
#define KASAN_OFFSET_ARM64      0xe000000000000000ULL
#endif  /* KASAN_OFFSET_ARM64 */

#if defined(ARM_LARGE_MEMORY)
#define KASAN_SHADOW_MIN        (VM_MAX_KERNEL_ADDRESS+1)
#define KASAN_SHADOW_MAX        0xffffffffffffffffULL
#else
#define KASAN_SHADOW_MIN        0xfffffffc00000000ULL
#define KASAN_SHADOW_MAX        0xffffffff80000000ULL
#endif

#endif /* _KASAN_CLASSIC_ARM64_H_ */
