/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 22, 2025.
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
#ifndef VPX_VPX_PORTS_ARM_H_
#define VPX_VPX_PORTS_ARM_H_
#include <stdlib.h>
#include "vpx_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Armv7-A optional Neon instructions, mandatory from Armv8.0-A.
#define HAS_NEON (1 << 0)
// Armv8.2-A optional Neon dot-product instructions, mandatory from Armv8.4-A.
#define HAS_NEON_DOTPROD (1 << 1)
// Armv8.2-A optional Neon i8mm instructions, mandatory from Armv8.6-A.
#define HAS_NEON_I8MM (1 << 2)
// Armv8.2-A optional SVE instructions, mandatory from Armv9.0-A.
#define HAS_SVE (1 << 3)
// Armv9.0-A SVE2 instructions.
#define HAS_SVE2 (1 << 4)

int arm_cpu_caps(void);

// Earlier gcc compilers have issues with some neon intrinsics
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ == 4 && \
    __GNUC_MINOR__ <= 6
#define VPX_INCOMPATIBLE_GCC
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_PORTS_ARM_H_
