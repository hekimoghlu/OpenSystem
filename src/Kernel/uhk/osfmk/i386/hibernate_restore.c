/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 6, 2023.
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
/*!
 * i386/x86_64-specific functions required to support hibernation resume.
 */

#include <i386/pmap.h>
#include <i386/proc_reg.h>
#include <IOKit/IOHibernatePrivate.h>
#include <vm/WKdm_new.h>

#include <machine/pal_hibernate.h>

extern pd_entry_t BootPTD[2048];

// src is virtually mapped, not page aligned,
// dst is a physical 4k page aligned ptr, len is one 4K page
// src & dst will not overlap

uintptr_t
hibernate_restore_phys_page(uint64_t src, uint64_t dst, uint32_t len, uint32_t procFlags)
{
	(void)procFlags;
	uint64_t * d;
	uint64_t * s;

	if (src == 0) {
		return (uintptr_t)dst;
	}

	d = (uint64_t *)pal_hib_map(DEST_COPY_AREA, dst);
	s = (uint64_t *) (uintptr_t)src;

	__nosan_memcpy(d, s, len);

	return (uintptr_t)d;
}
#undef hibprintf

void hibprintf(const char *fmt, ...);

uintptr_t
pal_hib_map(uintptr_t virt, uint64_t phys)
{
	uintptr_t index;

	switch (virt) {
	case DEST_COPY_AREA:
	case SRC_COPY_AREA:
	case COPY_PAGE_AREA:
	case BITMAP_AREA:
	case IMAGE_AREA:
	case IMAGE2_AREA:
	case SCRATCH_AREA:
	case WKDM_AREA:
		break;

	default:
		asm("cli;hlt;");
		break;
	}
	if (phys < WKDM_AREA) {
		// first 4Gb is all mapped,
		// and do not expect source areas to cross 4Gb
		return phys;
	}
	index = (virt >> I386_LPGSHIFT);
	virt += (uintptr_t)(phys & I386_LPGMASK);
	phys  = ((phys & ~((uint64_t)I386_LPGMASK)) | INTEL_PTE_PS  | INTEL_PTE_VALID | INTEL_PTE_WRITE);
	if (phys == BootPTD[index]) {
		return virt;
	}
	BootPTD[index] = phys;
	invlpg(virt);
	BootPTD[index + 1] = (phys + I386_LPGBYTES);
	invlpg(virt + I386_LPGBYTES);

	return virt;
}

void
pal_hib_restore_pal_state(uint32_t *arg)
{
	(void)arg;
}

void
pal_hib_resume_init(__unused pal_hib_ctx_t *ctx, __unused hibernate_page_list_t *map, __unused uint32_t *nextFree)
{
}

void
pal_hib_restored_page(__unused pal_hib_ctx_t *ctx, __unused pal_hib_restore_stage_t stage, __unused ppnum_t ppnum)
{
}

void
pal_hib_patchup(__unused pal_hib_ctx_t *ctx)
{
}

void
pal_hib_decompress_page(void *src, void *dst, void *scratch, unsigned int compressedSize)
{
	WKdm_decompress_new((WK_word*)src, (WK_word*)dst, (WK_word*)scratch, compressedSize);
}
