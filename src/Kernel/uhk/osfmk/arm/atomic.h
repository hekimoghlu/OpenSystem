/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 16, 2023.
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
#ifndef _MACHINE_ATOMIC_H
#error "Do not include <arm/atomic.h> directly, use <machine/atomic.h>"
#endif

#ifndef _ARM_ATOMIC_H_
#define _ARM_ATOMIC_H_

#include <mach/boolean.h>

// Parameter for __builtin_arm_dmb
#define DMB_OSHLD       0x1
#define DMB_OSHST       0x2
#define DMB_OSH         0x3
#define DMB_NSHLD       0x5
#define DMB_NSHST       0x6
#define DMB_NSH         0x7
#define DMB_ISHLD       0x9
#define DMB_ISHST       0xa
#define DMB_ISH         0xb
#define DMB_LD          0xd
#define DMB_ST          0xe
#define DMB_SY          0xf

// Parameter for __builtin_arm_dsb
#define DSB_OSHLD       0x1
#define DSB_OSHST       0x2
#define DSB_OSH         0x3
#define DSB_NSHLD       0x5
#define DSB_NSHST       0x6
#define DSB_NSH         0x7
#define DSB_ISHLD       0x9
#define DSB_ISHST       0xa
#define DSB_ISH         0xb
#define DSB_LD          0xd
#define DSB_ST          0xe
#define DSB_SY          0xf

// Parameter for __builtin_arm_isb
#define ISB_SY          0xf

#endif // _ARM_ATOMIC_H_
