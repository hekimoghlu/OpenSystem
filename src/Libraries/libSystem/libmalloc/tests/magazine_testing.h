/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 21, 2025.
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
#ifndef __MAGAZINE_TESTING
#define __MAGAZINE_TESTING

// Import the mvm_* functions for magazines to use
#import "../src/vm.c"

#import "../src/magazine_rack.c"

bool
malloc_tracing_enabled = false;

int
recirc_retained_regions = DEFAULT_RECIRC_RETAINED_REGIONS;

// Stub out cross-file dependencies so that they just assert.
void malloc_report(uint32_t flags, const char *fmt, ...)
{
	__builtin_trap();
}

void
malloc_zone_error(uint32_t flags, bool is_corruption, const char *fmt, ...)
{
	__builtin_trap();
}
void
malloc_zone_check_fail(const char *msg, const char *fmt, ...)
{
	__builtin_trap();
}

void
szone_free(szone_t *szone, void *ptr)
{
	__builtin_trap();
}

void *
szone_malloc(szone_t *szone, size_t size)
{
	__builtin_trap();
}

void
test_rack_setup(rack_t *rack, rack_type_t rack_type)
{
	memset(rack, 'a', sizeof(rack));
	rack_init(rack, rack_type, 1, 0);
	T_QUIET; T_ASSERT_NOTNULL(rack->magazines, "magazine initialisation");
}

#endif // __MAGAZINE_TESTING
