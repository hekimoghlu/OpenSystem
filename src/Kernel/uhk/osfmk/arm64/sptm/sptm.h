/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 11, 2025.
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
#pragma once

/**
 * This file is meant to be the main header that XNU uses to get access to all
 * of the exported SPTM types, declarations, and function prototypes. Wrappers
 * around some of the SPTM library functions are also located in here.
 */
#include <sptm/debug_header.h>
#include <sptm/sptm_xnu.h>
#include <kern/debug.h>

#include <stdbool.h>

/* Bootstrapping arguments passed from the SPTM to XNU. */
extern const sptm_bootstrap_args_xnu_t *SPTMArgs;

typedef struct arm_physrange {
	uint64_t        start_phys;     /* Starting physical address */
	uint64_t        end_phys;       /* Ending physical address (EXCLUSIVE) */
} arm_physrange_t;

/**
 * Convenience function for checking whether an SPTM operation on the given page
 * is in-flight.
 *
 * @note This is just a wrapper around the SPTM library.
 *
 * @param paddr The physical address of the managed page against which to check
 *              for in-flight operations.
 *
 * @return True if an operation is in-flight, false otherwise.
 */
static inline bool
sptm_paddr_is_inflight(sptm_paddr_t paddr)
{
	bool is_inflight = false;
	if (sptm_check_inflight(paddr, &is_inflight) != LIBSPTM_SUCCESS) {
		panic("%s: sptm_check_inflight returned failure for paddr 0x%llx",
		    __func__, (uint64_t)paddr);
	}

	return is_inflight;
}

/**
 * Convenience function for determining the SPTM frame type for a given
 * SPTM-managed page.
 *
 * @note This is just a wrapper around the SPTM library.
 *
 * @param paddr The physical address of the managed page to get the type of.
 *
 * @return The SPTM type for the given frame. If the page passed in is not an
 *         SPTM-managed page, then a panic will get triggered.
 */
static inline sptm_frame_type_t
sptm_get_frame_type(sptm_paddr_t paddr)
{
	sptm_frame_type_t frame_type;
	if (sptm_get_paddr_type(paddr, &frame_type) != LIBSPTM_SUCCESS) {
		panic("%s: sptm_get_paddr_type returned failure for paddr 0x%llx",
		    __func__, (uint64_t)paddr);
	}

	return frame_type;
}

/**
 * Convenience function for checking if a given SPTM-managed
 * page has any mappings.
 *
 * @note This is just a wrapper around the SPTM library.
 *
 * @param paddr The physical address of the managed page to query.
 *
 */
static inline bool
sptm_frame_is_last_mapping(sptm_paddr_t paddr, libsptm_refcnt_type_t refcnt_type)
{
	bool is_last;
	if (sptm_paddr_is_last_mapping(paddr, refcnt_type, &is_last) != LIBSPTM_SUCCESS) {
		panic("%s: sptm_paddr_is_last_mapping returned failure for paddr 0x%llx",
		    __func__, (uint64_t)paddr);
	}

	return is_last;
}

/**
 * Convenience function for retrieving the SPTM page table mapping reference
 * count.
 *
 * @note This is just a wrapper around the SPTM library.
 *
 * @param table_paddr The physical address of the page table page for which to
 *                    obtain the mapping reference count.
 *
 * @return The SPTM mapping reference count for the page table page.  If the page
 *         passed in is not an SPTM-managed page table page, then a panic will be
 *         triggered.
 */
static inline uint16_t
sptm_get_page_table_refcnt(sptm_paddr_t table_paddr)
{
	uint16_t refcnt;
	if (sptm_get_table_mapping_count(table_paddr, &refcnt) != LIBSPTM_SUCCESS) {
		panic("%s: sptm_get_table_mapping_count returned failure for paddr 0x%llx",
		    __func__, (uint64_t)table_paddr);
	}

	return refcnt;
}
