/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#ifndef _MACHINE_ARM64_KPC_H
#define _MACHINE_ARM64_KPC_H

#include <pexpert/arm64/board_config.h>
#include <stdint.h>

typedef uint64_t kpc_config_t;

#define KPC_ARM64_FIXED_COUNT        (2)
#define KPC_ARM64_CONFIGURABLE_COUNT (CORE_NCTRS - KPC_ARM64_FIXED_COUNT)

#if CPMU_64BIT_PMCS
#define KPC_ARM64_COUNTER_WIDTH    (63)
#define KPC_ARM64_COUNTER_OVF_BIT  (63)
#else // CPMU_64BIT_PMCS
#define KPC_ARM64_COUNTER_WIDTH    (47)
#define KPC_ARM64_COUNTER_OVF_BIT  (47)
#endif // !CPMU_64BIT_PMCS

#define KPC_ARM64_COUNTER_MASK     ((UINT64_C(1) << KPC_ARM64_COUNTER_WIDTH) - 1)
#define KPC_ARM64_COUNTER_OVF_MASK (UINT64_C(1) << KPC_ARM64_COUNTER_OVF_BIT)

/* arm64 uses fixed counter shadows */
#define FIXED_COUNTER_SHADOW (1)

#define KPC_ARM64_PMC_COUNT (KPC_ARM64_FIXED_COUNT + KPC_ARM64_CONFIGURABLE_COUNT)

/* Size to the maximum number of counters we could read from every class in one go */
#define KPC_MAX_COUNTERS (KPC_ARM64_FIXED_COUNT + KPC_ARM64_CONFIGURABLE_COUNT + 1)

#endif /* _MACHINE_ARM64_KPC_H */
