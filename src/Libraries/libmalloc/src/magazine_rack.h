/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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
#ifndef __MAGAZINE_RACK_H
#define __MAGAZINE_RACK_H

#include <malloc/_ptrcheck.h>
__ptrcheck_abi_assume_single()

/*******************************************************************************
 * Definitions for region hash
 ******************************************************************************/

typedef void * __single region_t;
typedef region_t * __single rgnhdl_t; /* A pointer into hashed_regions array. */
typedef struct region_trailer region_trailer_t;

#define INITIAL_NUM_REGIONS_SHIFT 6							 // log2(INITIAL_NUM_REGIONS)
#define INITIAL_NUM_REGIONS (1 << INITIAL_NUM_REGIONS_SHIFT) // Must be a power of 2!
#define HASHRING_OPEN_ENTRY ((region_t)0)					 // Initial value and sentinel marking end of collision chain
#define HASHRING_REGION_DEALLOCATED __unsafe_forge_single(void *, ~0UL)			 // Region at this slot reclaimed by OS
#define HASH_BLOCKS_ALIGN TINY_BLOCKS_ALIGN					 // MIN( TINY_BLOCKS_ALIGN, SMALL_BLOCKS_ALIGN, ... )

typedef struct region_hash_generation {
	size_t num_regions_allocated;
	size_t num_regions_allocated_shift; // log2(num_regions_allocated)
	region_t * __counted_by(num_regions_allocated) hashed_regions;			// hashed by location
	struct region_hash_generation *nextgen;
} region_hash_generation_t;

OS_ENUM(rack_type, uint32_t,
	RACK_TYPE_NONE = 0,
	RACK_TYPE_TINY,
	RACK_TYPE_SMALL,
	RACK_TYPE_MEDIUM,
);

/*******************************************************************************
 * Per-allocator collection of regions and magazines
 ******************************************************************************/

typedef struct rack_s {
	/* Regions for tiny objects */
	_malloc_lock_s region_lock MALLOC_CACHE_ALIGN;

	rack_type_t type;
	size_t num_regions;
	size_t num_regions_dealloc;
	region_hash_generation_t *region_generation;
	region_hash_generation_t rg[2];
	region_t initial_regions[INITIAL_NUM_REGIONS];

	int num_magazines;
	unsigned num_magazines_mask;
	int num_magazines_mask_shift;
	uint32_t debug_flags;

	// array of per-processor magazines
	magazine_t *magazines;

	uintptr_t cookie;
	uintptr_t last_madvise;
} rack_t;

MALLOC_NOEXPORT
void
rack_init(rack_t *rack, rack_type_t type, uint32_t num_magazines, uint32_t debug_flags);

MALLOC_NOEXPORT
void
rack_destroy_regions(rack_t *rack, size_t region_size);

MALLOC_NOEXPORT
void
rack_destroy(rack_t *rack);

MALLOC_NOEXPORT
void
rack_region_insert(rack_t *rack, region_t region);

MALLOC_NOEXPORT
bool
rack_region_remove(rack_t *rack, region_t region, region_trailer_t *trailer);

MALLOC_NOEXPORT
bool
rack_region_maybe_dispose(rack_t *rack, region_t region, size_t region_size,
		region_trailer_t *trailer);

MALLOC_NOEXPORT
unsigned int
rack_get_thread_index(rack_t *rack);

MALLOC_NOEXPORT MALLOC_ALWAYS_INLINE MALLOC_INLINE
static void
rack_region_lock(rack_t *rack)
{
	_malloc_lock_lock(&rack->region_lock);
}

MALLOC_NOEXPORT MALLOC_ALWAYS_INLINE MALLOC_INLINE
static void
rack_region_unlock(rack_t *rack)
{
	_malloc_lock_unlock(&rack->region_lock);
}

#endif // __MAGAZINE_RACK_H
