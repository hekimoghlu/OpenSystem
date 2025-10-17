/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 9, 2021.
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
#ifndef __NANO_ZONE_COMMON_H
#define __NANO_ZONE_COMMON_H

#define NANO_MAX_SIZE			256 /* Buckets sized {16, 32, 48, ..., 256} */
#define SHIFT_NANO_QUANTUM		4
#define NANO_REGIME_QUANTA_SIZE	(1 << SHIFT_NANO_QUANTUM)	// 16
#define NANO_QUANTA_MASK		(NANO_REGIME_QUANTA_SIZE - 1)
#define NANO_SIZE_CLASSES		(NANO_MAX_SIZE/NANO_REGIME_QUANTA_SIZE)

#if MALLOC_TARGET_IOS

// Nanozone follows the shared region.
#define SHIFT_NANO_SIGNATURE	29
#define NANOZONE_SIGNATURE_BITS	35
#define NANOZONE_BASE_REGION_ADDRESS (SHARED_REGION_BASE + SHARED_REGION_SIZE)
#define NANOZONE_SIGNATURE (NANOZONE_BASE_REGION_ADDRESS >> SHIFT_NANO_SIGNATURE)

#else // MALLOC_TARGET_IOS

#define SHIFT_NANO_SIGNATURE	44
#define NANOZONE_SIGNATURE_BITS	20
#define NANOZONE_SIGNATURE		0x6ULL
#define NANOZONE_BASE_REGION_ADDRESS (NANOZONE_SIGNATURE << SHIFT_NANO_SIGNATURE)

#endif // MALLOC_TARGET_IOS


MALLOC_STATIC_ASSERT((SHIFT_NANO_SIGNATURE + NANOZONE_SIGNATURE_BITS) == 64, 
		"Nano addresses must be 64 bits wide");

static MALLOC_INLINE size_t
_nano_common_good_size(size_t size)
{
	return (size <= NANO_REGIME_QUANTA_SIZE) ? NANO_REGIME_QUANTA_SIZE
		: (((size + NANO_REGIME_QUANTA_SIZE - 1) >> SHIFT_NANO_QUANTUM) << SHIFT_NANO_QUANTUM);
}

#endif // __NANO_ZONE_COMMON_H
