/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 7, 2023.
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
#ifndef _PEXPERT_ARM64_VMAPPLE_H
#define _PEXPERT_ARM64_VMAPPLE_H

#define NO_MONITOR                1
#define NO_ECORE                  1

#define VMAPPLE                   1
#define APPLEVIRTUALPLATFORM      1

#define CPU_HAS_APPLE_PAC         1
#define HAS_PARAVIRTUALIZED_PAC   1
#define HAS_GIC_V3                1
#define HAS_ARM_FEAT_SSBS2        1
#define HAS_ARM_FEAT_SME          1
#define HAS_ARM_FEAT_SME2         1
#define HAS_ARM_FEAT_PAN3         1

#define __ARM_PAN_AVAILABLE__     1
#define __ARM_16K_PG__            1
#define __ARM_RANGE_TLBI__        1

#define ARM_PARAMETERIZED_PMAP    1
#define __ARM_MIXED_PAGE_SIZE__   1

#include <pexpert/arm64/apple_arm64_common.h>
#undef  __ARM64_PMAP_SUBPAGE_L1__

#ifndef ASSEMBLER
#define PL011_UART
#define PLATFORM_PANIC_LOG_DISABLED
#endif /* ! ASSEMBLER */


#define GIC_SPURIOUS_IRQ          1023    // IRQ no. for GIC spurious interrupt

#define GICR_PE_SIZE              0x20000 // Size of each redistributor region


/* GICv3 reigster definitions; see GICv3 spec (Arm IHI 0069G) for more about these registers */
#define GICD_CTLR                 0x0

#define GICD_CTLR_ENABLEGRP0      0x1

#define GICR_TYPER                              0x08
#define GICR_WAKER                              0x14
#define GICR_IGROUPR0                           0x10080
#define GICR_ISENABLER0                         0x10100

#define GICR_TYPER_AFFINITY_VALUE_SHIFT         32
#define GICR_TYPER_LAST                         0x10

#define GICR_WAKER_PROCESSORSLEEP               0x2
#define GICR_WAKER_CHILDRENASLEEP               0x4

#define ICC_CTLR_EOIMODE                        0x1

#define ICC_SRE_SRE                             0x1
/* End of GICv3 register definitions */

#endif /* ! _PEXPERT_ARM64_VMAPPLE_H */
