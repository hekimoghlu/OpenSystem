/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 2, 2025.
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
 *	File: mach/shared_memory_server.h
 *
 *      protos and struct definitions for shared library
 *	server and interface
 */

/*
 * XXX
 *
 * NOTE: this file is deprecated and will be removed in the near future.
 * Any project that includes this file should be changed to:
 * 1. use <mach/shared_region.h> instead of this file,
 * 2. handle the new shared regions, now available on more platforms
 */

#ifndef _MACH_SHARED_MEMORY_SERVER_H_
#define _MACH_SHARED_MEMORY_SERVER_H_

#warning "<mach/shared_memory_server.h> is deprecated.  Please use <mach/shared_region.h> instead."

#include <sys/cdefs.h>
#include <mach/vm_prot.h>
#include <mach/vm_types.h>
#include <mach/mach_types.h>

#define VM_PROT_COW  0x8  /* must not interfere with normal prot assignments */
#define VM_PROT_ZF  0x10  /* must not interfere with normal prot assignments */

#ifdef  __arm__
#define GLOBAL_SHARED_TEXT_SEGMENT      0x30000000U
#define GLOBAL_SHARED_DATA_SEGMENT      0x38000000U
#define GLOBAL_SHARED_SEGMENT_MASK      0xF8000000U

#define SHARED_TEXT_REGION_SIZE         0x08000000
#define SHARED_DATA_REGION_SIZE         0x08000000
#else
#define GLOBAL_SHARED_TEXT_SEGMENT      0x90000000U
#define GLOBAL_SHARED_DATA_SEGMENT      0xA0000000U
#define GLOBAL_SHARED_SEGMENT_MASK      0xF0000000U

#define SHARED_TEXT_REGION_SIZE         0x10000000
#define SHARED_DATA_REGION_SIZE         0x10000000
#endif

#if !defined(__LP64__)

#define SHARED_LIBRARY_SERVER_SUPPORTED

#define SHARED_ALTERNATE_LOAD_BASE      0x09000000

/*
 *  Note: the two masks below are useful because the assumption is
 *  made that these shared regions will always be mapped on natural boundaries
 *  i.e. if the size is 0x10000000 the object can be mapped at
 *  0x20000000, or 0x30000000, but not 0x1000000
 */
#ifdef  __arm__
#define SHARED_TEXT_REGION_MASK         0x07FFFFFF
#define SHARED_DATA_REGION_MASK         0x07FFFFFF
#else
#define SHARED_TEXT_REGION_MASK         0x0FFFFFFF
#define SHARED_DATA_REGION_MASK         0x0FFFFFFF
#endif


/* flags field aliases for copyin_shared_file and load_shared_file */

/* IN */
#define ALTERNATE_LOAD_SITE 0x1
#define NEW_LOCAL_SHARED_REGIONS 0x2
#define QUERY_IS_SYSTEM_REGION 0x4

/* OUT */
#define SF_PREV_LOADED    0x1
#define SYSTEM_REGION_BACKED 0x2


struct sf_mapping {
	vm_offset_t     mapping_offset;
	vm_size_t       size;
	vm_offset_t     file_offset;
	vm_prot_t       protection;  /* read/write/execute/COW/ZF */
	vm_offset_t     cksum;
};
typedef struct sf_mapping sf_mapping_t;

#endif  /* !defined(__LP64__) */

/*
 * All shared_region_* declarations are a private interface
 * between dyld and the kernel.
 *
 */
struct shared_region_range_np {
	mach_vm_address_t       srr_address;
	mach_vm_size_t          srr_size;
};

#ifndef KERNEL

__BEGIN_DECLS
int     shared_region_map_file_np(int fd,
    uint32_t mappingCount,
    const struct shared_file_mapping_np *mappings,
    int64_t *slide_p);
int     shared_region_make_private_np(uint32_t rangeCount,
    const struct shared_region_range_np *ranges);
__END_DECLS

#endif /* !KERNEL */

#endif /* _MACH_SHARED_MEMORY_SERVER_H_ */
