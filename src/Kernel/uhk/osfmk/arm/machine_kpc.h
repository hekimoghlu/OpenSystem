/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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
#ifndef _MACHINE_ARM_KPC_H
#define _MACHINE_ARM_KPC_H

#include <stdint.h>

#ifdef ARMA7

#define KPC_ARM_FIXED_COUNT             1
#define KPC_ARM_CONFIGURABLE_COUNT      4

#define KPC_ARM_TOTAL_COUNT                     (KPC_ARM_FIXED_COUNT + KPC_ARM_CONFIGURABLE_COUNT)

#define KPC_ARM_COUNTER_WIDTH 32

#else

#define KPC_ARM_FIXED_COUNT             2
#define KPC_ARM_CONFIGURABLE_COUNT      6

#define KPC_ARM_COUNTER_WIDTH 39
#define KPC_ARM_COUNTER_MASK ((1ull << KPC_ARM_COUNTER_WIDTH) - 1)
#define KPC_ARM_COUNTER_OVF_BIT (39)
#define KPC_ARM_COUNTER_OVF_MASK (1ull << KPC_ARM_COUNTER_OVF_BIT)

#endif

typedef uint64_t kpc_config_t;

/* Size to the maximum number of counters we could read from every class in one go */
#define KPC_MAX_COUNTERS (KPC_ARM_FIXED_COUNT + KPC_ARM_CONFIGURABLE_COUNT + 1)

/* arm32 uses fixed counter shadows */
#define FIXED_COUNTER_SHADOW  (1)

#endif /* _MACHINE_ARM_KPC_H */
