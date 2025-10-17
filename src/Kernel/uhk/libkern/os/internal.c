/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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
#include "trace_internal.h"
#include <mach-o/loader.h>
#include <string.h>

static bool
_os_trace_addr_in_text_segment_32(const void *dso, const void *addr)
{
	const struct mach_header *mhp = (const struct mach_header *) dso;
	const struct segment_command *sgp = (const struct segment_command *)(const void *)((const char *)mhp + sizeof(struct mach_header));

	for (uint32_t i = 0; i < mhp->ncmds; i++) {
		if (sgp->cmd == LC_SEGMENT) {
			if (strncmp(sgp->segname, SEG_TEXT, sizeof(sgp->segname)) == 0) {
				return (uintptr_t)addr >= (sgp->vmaddr) && (uintptr_t)addr < (sgp->vmaddr + sgp->vmsize);
			}
		}
		sgp = (const struct segment_command *)(const void *)((const char *)sgp + sgp->cmdsize);
	}

	return false;
}

static bool
_os_trace_addr_in_text_segment_64(const void *dso, const void *addr)
{
	const struct mach_header_64 *mhp = (const struct mach_header_64 *) dso;
	const struct segment_command_64 *sgp = (const struct segment_command_64 *)(const void *)((const char *)mhp + sizeof(struct mach_header_64));

	for (uint32_t i = 0; i < mhp->ncmds; i++) {
		if (sgp->cmd == LC_SEGMENT_64) {
			if (strncmp(sgp->segname, SEG_TEXT, sizeof(sgp->segname)) == 0) {
				return (uintptr_t)addr >= (sgp->vmaddr) && (uintptr_t)addr < (sgp->vmaddr + sgp->vmsize);
			}
		}
		sgp = (const struct segment_command_64 *)(const void *)((const char *)sgp + sgp->cmdsize);
	}

	return false;
}

bool
_os_trace_addr_in_text_segment(const void *dso, const void *addr)
{
	const struct mach_header *mhp = (const struct mach_header *) dso;
	bool retval = false;

	switch (mhp->magic) {
	case MH_MAGIC:
		retval = _os_trace_addr_in_text_segment_32(dso, addr);
		break;

	case MH_MAGIC_64:
		retval = _os_trace_addr_in_text_segment_64(dso, addr);
		break;

	default:
		retval = false;
		break;
	}

	return retval;
}
