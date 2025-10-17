/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 29, 2025.
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
#ifndef _MACHINE_X86_64_KPC_H
#define _MACHINE_X86_64_KPC_H

#include <stdint.h>

/* x86 config registers are 64-bit */
typedef uint64_t kpc_config_t;

/* Size to the maximum number of counters we could read from every
 * class in one go
 */
#define KPC_MAX_COUNTERS (32)

/* number of fixed config registers on x86_64 */
#define KPC_X86_64_FIXED_CONFIGS (1)

#endif /* _MACHINE_X86_64_KPC_H */
