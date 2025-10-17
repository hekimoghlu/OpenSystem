/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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
/*
 *
 *	File: mach/dyld_pager.h
 *
 *      protos and struct definitions for the pager that applies dyld fixups.
 */

#ifndef _MACH_DYLD_PAGER_H_
#define _MACH_DYLD_PAGER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <mach/vm_prot.h>
#include <mach/mach_types.h>

/*
 * These describe the address regions (mwlr_address, mwlr_size) to be mapped
 * from the given file (mwlr_fd, mwlr_file_offset) with mwlr_protections.
 */
struct mwl_region {
	int                  mwlr_fd;      /* fd of file file to over map */
	vm_prot_t            mwlr_protections;/* protections for new overmapping */
	uint64_t             mwlr_file_offset;/* offset in file of start of mapping */
	mach_vm_address_t    mwlr_address __kernel_data_semantics; /* start address of existing region */
	mach_vm_size_t       mwlr_size;    /* size of existing region */
};

#define MWL_INFO_VERS 7
struct mwl_info_hdr {
	uint32_t        mwli_version;            /* version of info blob, currently 7 */
	uint16_t        mwli_page_size;          /* 0x1000 or 0x4000 (for sanity checking) */
	uint16_t        mwli_pointer_format;     /* DYLD_CHAINED_PTR_* value */
	uint32_t        mwli_binds_offset;       /* offset within this blob of bind pointers table */
	uint32_t        mwli_binds_count;        /* number of pointers in bind pointers table (for range checks) */
	uint32_t        mwli_chains_offset;      /* offset within this blob of dyld_chained_starts_in_image */
	uint32_t        mwli_chains_size;        /* size of dyld_chained_starts_in_image */
	uint64_t        mwli_slide;              /* slide to add to rebased pointers */
	uint64_t        mwli_image_address;      /* add this to rebase offsets includes any slide */
	/* followed by the binds pointers and dyld_chained_starts_in_image */
};

#define MWL_MAX_REGION_COUNT 5  /* data, const, data auth, auth const, objc const */

#ifndef KERNEL_PRIVATE

extern int __map_with_linking_np(const struct mwl_region regions[], uint32_t regionCount, const struct mwl_info_hdr* blob, uint32_t blobSize);

#endif /* KERNEL_PRIVATE */

/*
 * Special value for dyld to use with shared_region_check_np() to prevent anymore use of map_with_linking_np() in a process
 */
#define DYLD_VM_END_MWL (-1ull)

#ifdef __cplusplus
}
#endif

#endif /* _MACH_DYLD_PAGER_H_ */
