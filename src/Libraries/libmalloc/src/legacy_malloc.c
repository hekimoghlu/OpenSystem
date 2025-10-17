/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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

/*
 * For use by CheckFix: create a new zone whose behavior is, apart from
 * the use of death-row and per-CPU magazines, that of Leopard.
 */
static MALLOC_NOINLINE void *
legacy_valloc(szone_t *szone, size_t size)
{
	void *ptr;
	size_t num_kernel_pages;

	num_kernel_pages = round_large_page_quanta(size) >> large_vm_page_quanta_shift;
	ptr = large_malloc(szone, num_kernel_pages, 0, TRUE);
#if DEBUG_MALLOC
	if (LOG(szone, ptr)) {
		malloc_report(ASL_LEVEL_INFO, "legacy_valloc returned %p\n", ptr);
	}
#endif
	return ptr;
}

malloc_zone_t *
create_legacy_scalable_zone(size_t initial_size, unsigned debug_flags)
{
	malloc_zone_t *mzone = create_scalable_zone(initial_size, debug_flags);
	szone_t *szone = (szone_t *)mzone;

	if (!szone) {
		return NULL;
	}

	mprotect(szone, sizeof(szone->basic_zone), PROT_READ | PROT_WRITE);
	szone->basic_zone.valloc = (void *)legacy_valloc;
	szone->basic_zone.free_definite_size = NULL;
	mprotect(szone, sizeof(szone->basic_zone), PROT_READ);

	return mzone;
}
