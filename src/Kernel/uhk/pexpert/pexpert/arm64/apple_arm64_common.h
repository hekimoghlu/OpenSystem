/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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
#ifndef _PEXPERT_ARM64_APPLE_ARM64_COMMON_H
#define _PEXPERT_ARM64_APPLE_ARM64_COMMON_H

#define __ARM_ARCH__                         8
#define __ARM_VMSA__                         8
#define __ARM_VFP__                          4
#define __ARM_COHERENT_CACHE__               1
#define __ARM_COHERENT_IO__                  1
#define __ARM_IC_NOALIAS_ICACHE__            1
#define __ARM_DEBUG__                        7
#define __ARM_ENABLE_SWAP__                  1
#define __ARM_V8_CRYPTO_EXTENSIONS__         1

#ifndef ARM_LARGE_MEMORY
#define __ARM64_PMAP_SUBPAGE_L1__            1
#endif

#define APPLE_ARM64_ARCH_FAMILY              1
#define ARM_ARCH_TIMER

#if defined(HAS_CTRR3)
#define KERNEL_INTEGRITY_CTRR                1
#define KERNEL_CTRR_VERSION                  3
#elif defined(HAS_CTRR)
#define KERNEL_INTEGRITY_CTRR                1
#define KERNEL_CTRR_VERSION                  2
#elif defined(HAS_KTRR)
#define KERNEL_INTEGRITY_KTRR                1
#elif defined(MONITOR)
#define KERNEL_INTEGRITY_WT                  1
#endif

#if defined(CPU_HAS_APPLE_PAC) && defined(__arm64e__)
#define HAS_APPLE_PAC                        1 /* Has Apple ARMv8.3a pointer authentication */
#endif

#include <pexpert/arm64/apple_arm64_regs.h>
#include <pexpert/arm64/apple_arm64_cpu.h>
#include <pexpert/arm64/AIC.h>

#ifndef ASSEMBLER
#include <pexpert/arm/apple_uart_regs.h>

#if !defined(APPLETYPHOON) && !defined(APPLETWISTER) && !defined(APPLEVIRTUALPLATFORM)
#include <pexpert/arm/dockchannel.h>

// AOP_CLOCK frequency * 30 ms
#define DOCKCHANNEL_DRAIN_PERIOD             (192000000 * 0.03)
#endif

#endif /* ASSEMBLER */

/*
 * See arm64/proc_reg.h for how these values are constructed from the MIDR.
 * The chip-revision property from EDT also uses these constants.
 */
#define CPU_VERSION_A0                       0x00
#define CPU_VERSION_A1                       0x01
#define CPU_VERSION_B0                       0x10
#define CPU_VERSION_B1                       0x11
#define CPU_VERSION_C0                       0x20
#define CPU_VERSION_UNKNOWN                  0xff


/*
 * Conservatively assume that BTI will be enforced.
 * Individual SoCs and kernel configurations may have different behavior.
 */
#define BTI_ENFORCED 1

#endif /* !_PEXPERT_ARM64_APPLE_ARM64_COMMON_H */
