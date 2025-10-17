/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#include <pexpert/pexpert.h>
#include <pexpert/arm/consistent_debug.h>
#include <pexpert/device_tree.h>
#include <libkern/OSAtomic.h>
#include <machine/machine_routines.h>

static dbg_registry_t * consistent_debug_registry = NULL;

static dbg_record_header_t*
consistent_debug_allocate_entry(void)
{
	unsigned int i;

	if (!consistent_debug_registry) {
		return NULL;
	}
	for (i = 0; i < consistent_debug_registry->top_level_header.num_records; i++) {
		dbg_record_header_t *record = &consistent_debug_registry->records[i];
		if (OSCompareAndSwap64(kDbgIdUnusedEntry, kDbgIdReservedEntry, &record->record_id)) {
			// Reserved an entry at position i.
			return (dbg_record_header_t*)record;
		}
	}
	return NULL;
}

boolean_t
PE_consistent_debug_lookup_entry(uint64_t record_id, uint64_t *phys_addr, uint64_t *length)
{
	assert(phys_addr != NULL);
	assert(length != NULL);

	for (unsigned int i = 0; i < consistent_debug_registry->top_level_header.num_records; i++) {
		if (consistent_debug_registry->records[i].record_id == record_id) {
			*phys_addr = consistent_debug_registry->records[i].physaddr;
			*length = consistent_debug_registry->records[i].length;

			return true;
		}
	}

	return false;
}

int
PE_consistent_debug_inherit(void)
{
	DTEntry         entryP;
	uintptr_t const *prop_data;
	uintptr_t       root_pointer = 0;
	uint32_t        size;

	if ((SecureDTLookupEntry(NULL, "/chosen", &entryP) == kSuccess)) {
		if (SecureDTGetProperty(entryP, "consistent-debug-root", (void const **)&prop_data, &size) == kSuccess) {
			root_pointer = prop_data[0];
		}
	}
	if (root_pointer == 0) {
		return -1;
	}
	consistent_debug_registry = (dbg_registry_t *)ml_map_high_window(root_pointer, sizeof(dbg_registry_t));
	return 0;
}

int
PE_consistent_debug_register(uint64_t record_id, uint64_t physaddr, uint64_t length)
{
	dbg_record_header_t *allocated_header = consistent_debug_allocate_entry();
	if (allocated_header == NULL) {
		return -1;
	}
	allocated_header->length = length;
	allocated_header->physaddr = physaddr;
	// Make sure the hdr/length are visible before the record_id.
	__asm__ volatile ("dmb ish" : : : "memory");
	allocated_header->record_id = record_id;
	return 0;
}

int
PE_consistent_debug_enabled(void)
{
	return consistent_debug_registry != NULL;
}
