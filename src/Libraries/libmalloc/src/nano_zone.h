/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 24, 2024.
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
#ifndef __NANO_ZONE_H
#define __NANO_ZONE_H

#if CONFIG_NANOZONE

/*********************	DEFINITIONS	************************/

#define MAX_RECORDER_BUFFER		256

/*************          nanozone address field layout        ******************/

#if defined(__x86_64)
#define NANO_MAG_BITS			6
#define NANO_BAND_BITS			17
#define NANO_SLOT_BITS			4
#define NANO_OFFSET_BITS		17

#else
#error Unknown Architecture
#endif

// clang-format really dislikes the bitfields here
// clang-format off
#if defined(__BIG_ENDIAN__)
struct nano_blk_addr_s {
    uint64_t
	nano_signature:NANOZONE_SIGNATURE_BITS,	// the address range devoted to us.
	nano_mag_index:NANO_MAG_BITS,		// the core that allocated this block
	nano_band:NANO_BAND_BITS,
	nano_slot:NANO_SLOT_BITS,		// bucket of homogenous quanta-multiple blocks
	nano_offset:NANO_OFFSET_BITS;		// locates the block
};
#else
// least significant bits declared first
struct nano_blk_addr_s {
    uint64_t
	nano_offset:NANO_OFFSET_BITS,		// locates the block
	nano_slot:NANO_SLOT_BITS,		// bucket of homogenous quanta-multiple blocks
	nano_band:NANO_BAND_BITS,
	nano_mag_index:NANO_MAG_BITS,		// the core that allocated this block
	nano_signature:NANOZONE_SIGNATURE_BITS;	// the address range devoted to us.
};
#endif
// clang-format on

typedef union  {
    uint64_t			addr;
    struct nano_blk_addr_s	fields;
} nano_blk_addr_t;

#define SLOT_IN_BAND_SIZE 	(1 << NANO_OFFSET_BITS)
#define SLOT_KEY_LIMIT 		(1 << NANO_SLOT_BITS) /* Must track nano_slot width */
#define BAND_SIZE 		(1 << (NANO_SLOT_BITS + NANO_OFFSET_BITS)) /*  == Number of bytes covered by a page table entry */
#define NANO_MAG_SIZE 		(1 << NANO_MAG_BITS)
#define NANO_SLOT_SIZE 		(1 << NANO_SLOT_BITS)

#ifdef __INTERNAL_H

/****************************** zone itself ***********************************/

/*
 * Note that objects whose adddress are held in pointers here must be pursued
 * individually in the nano_in_use_enumeration() routines.
 */

typedef struct chained_block_s {
    uintptr_t			double_free_guard;
    struct chained_block_s	*next;
} *chained_block_t;

typedef struct nano_meta_s {
    OSQueueHead			slot_LIFO MALLOC_NANO_CACHE_ALIGN;
    unsigned int		slot_madvised_log_page_count;
    volatile uintptr_t		slot_current_base_addr;
    volatile uintptr_t		slot_limit_addr;
    volatile size_t		slot_objects_mapped;
    volatile size_t		slot_objects_skipped;
    bitarray_t			slot_madvised_pages;
    // position on cache line distinct from that of slot_LIFO
    volatile uintptr_t		slot_bump_addr MALLOC_NANO_CACHE_ALIGN;
    volatile boolean_t		slot_exhausted;
    unsigned int		slot_bytes;
    unsigned int		slot_objects;
} *nano_meta_admin_t;

// vm_allocate()'d, so page-aligned to begin with.
typedef struct nanozone_s {
    // first page will be given read-only protection
    malloc_zone_t		basic_zone;
    uint8_t			pad[PAGE_MAX_SIZE - sizeof(malloc_zone_t)];

    // remainder of structure is R/W (contains no function pointers)
    // page-aligned
    // max: NANO_MAG_SIZE cores x NANO_SLOT_SIZE slots for nano blocks {16 .. 256}
    struct nano_meta_s		meta_data[NANO_MAG_SIZE][NANO_SLOT_SIZE];
    _malloc_lock_s			band_resupply_lock[NANO_MAG_SIZE];
    uintptr_t           band_max_mapped_baseaddr[NANO_MAG_SIZE];
    size_t			core_mapped_size[NANO_MAG_SIZE];

    unsigned			debug_flags;

    /* security cookie */
    uintptr_t			cookie;

    /*
     * The nano zone constructed by create_nano_zone() would like to hand off tiny, small, and large
     * allocations to the default scalable zone. Record the latter as the "helper" zone here.
     */
    malloc_zone_t		*helper_zone;
} nanozone_t;

#define NANOZONE_PAGED_SIZE	((sizeof(nanozone_t) + vm_page_size - 1) & ~ (vm_page_size - 1))

#endif // __INTERNAL_H

#endif // CONFIG_NANOZONE

#endif // __NANO_ZONE_H
