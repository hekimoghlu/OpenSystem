/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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
#include "internal.h"


// Retrieve PAC-stripped address of introspection struct.  We avoid
// authenticating the pointer when loading it's value, because we can't
// authenticate pointers copied from remote processes (or corpses).
static vm_address_t
get_introspection_addr(const malloc_zone_t *zone)
{
	// return zone->malloc_zone.introspect;  // but without ptrauth
	vm_address_t ptr_addr =
			(vm_address_t)zone + offsetof(malloc_zone_t, introspect);
	vm_address_t addr = *(vm_address_t *)ptr_addr;
	return (vm_address_t)ptrauth_strip((malloc_introspection_t *)addr,
			ptrauth_key_process_independent_data);
}

kern_return_t
get_zone_type(task_t task, memory_reader_t reader,
		vm_address_t zone_address, unsigned *zone_type)
{
	MALLOC_ASSERT(reader);

	kern_return_t kr;
	*zone_type = MALLOC_ZONE_TYPE_UNKNOWN;

	// malloc_introspection_t::zone_type requires zone version >= 14
	malloc_zone_t *zone;
	kr = reader(task, zone_address, sizeof(malloc_zone_t), (void **)&zone);
	if (kr != KERN_SUCCESS) {
		return kr;
	}
	if (zone->version < 14) {
		return KERN_SUCCESS;
	}

	// Retrieve zone type
	vm_address_t zone_type_addr = get_introspection_addr(zone) +
			offsetof(malloc_introspection_t, zone_type);
	unsigned *zt;
	kr = reader(task, zone_type_addr, sizeof(unsigned), (void **)&zt);
	if (kr != KERN_SUCCESS) {
		return kr;
	}

	*zone_type = *zt;
	return KERN_SUCCESS;
}

kern_return_t
malloc_get_wrapped_zone(task_t task, memory_reader_t reader,
		vm_address_t zone_address, vm_address_t *wrapped_zone_address)
{
	reader = reader_or_in_memory_fallback(reader, task);

	kern_return_t kr;
	*wrapped_zone_address = (vm_address_t)NULL;

	unsigned zone_type;
	kr = get_zone_type(task, reader, zone_address, &zone_type);
	if (kr != KERN_SUCCESS) {
		return kr;
	}
	if (zone_type != MALLOC_ZONE_TYPE_PGM &&
			zone_type != MALLOC_ZONE_TYPE_SANITIZER) {
		return KERN_SUCCESS;
	}

	// Load wrapped zone address
	vm_address_t wrapped_zone_ptr_addr = zone_address + WRAPPED_ZONE_OFFSET;
	vm_address_t *wrapped_zone_addr;
	kr = reader(task, wrapped_zone_ptr_addr, sizeof(vm_address_t),
			(void **)&wrapped_zone_addr);
	if (kr != KERN_SUCCESS) {
		return kr;
	}

	*wrapped_zone_address = *wrapped_zone_addr;
	return KERN_SUCCESS;
}

// Internal, in-process helper for task-based SPI
malloc_zone_t *
get_wrapped_zone(malloc_zone_t *zone)
{
	malloc_zone_t *wrapped_zone;
	kern_return_t kr = malloc_get_wrapped_zone(mach_task_self(),
			/*memory_reader=*/NULL, (vm_address_t)zone, (vm_address_t *)&wrapped_zone);
	MALLOC_ASSERT(kr == KERN_SUCCESS);  // In-process lookup cannot fail
	return wrapped_zone;
}
