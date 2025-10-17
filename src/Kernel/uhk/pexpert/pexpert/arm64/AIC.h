/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 3, 2024.
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
#ifndef _PEXPERT_ARM_AIC_H
#define _PEXPERT_ARM_AIC_H

#ifndef ASSEMBLER

#include <stdint.h>

static inline uint32_t
_aic_read32(uintptr_t addr)
{
	return *(volatile uint32_t *)addr;
}

static inline void
_aic_write32(uintptr_t addr, uint32_t data)
{
	*(volatile uint32_t *)(addr) = data;
}

#define aic_read32(offset, data) (_aic_read32(pic_base + (offset)))
#define aic_write32(offset, data) (_aic_write32(pic_base + (offset), (data)))

#endif

// AIC timebase registers (timer base address in DT node is setup as AIC_BASE + 0x1000)
#define kAICMainTimLo                           (0x20)
#define kAICMainTimHi                           (0x28)

#endif /* ! _PEXPERT_ARM_AIC_H */
