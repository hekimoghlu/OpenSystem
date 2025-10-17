/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 30, 2025.
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
 *	File: vm/vm_shared_region.h
 *
 *      protos and struct definitions for shared region
 */

#ifndef _VM_SHARED_REGION_H_
#define _VM_SHARED_REGION_H_

#ifdef  KERNEL_PRIVATE

#include <mach/vm_prot.h>
#include <mach/mach_types.h>
#include <mach/shared_region.h>

#include <kern/kern_types.h>
#include <kern/macro_help.h>

#include <vm/vm_map.h>

extern int shared_region_version;
extern int shared_region_persistence;

#if DEBUG
extern int shared_region_debug;
#define SHARED_REGION_DEBUG(args)               \
	MACRO_BEGIN                             \
	if (shared_region_debug) {              \
	        kprintf args;                   \
	}                                       \
	MACRO_END
#else /* DEBUG */
#define SHARED_REGION_DEBUG(args)
#endif /* DEBUG */

extern int shared_region_trace_level;

extern struct vm_shared_region *primary_system_shared_region;

#define SHARED_REGION_TRACE_NONE_LVL            0 /* no trace */
#define SHARED_REGION_TRACE_ERROR_LVL           1 /* trace abnormal events */
#define SHARED_REGION_TRACE_INFO_LVL            2 /* trace all events */
#define SHARED_REGION_TRACE_DEBUG_LVL           3 /* extra traces for debug */
#define SHARED_REGION_TRACE(level, args)                \
	MACRO_BEGIN                                     \
	if (shared_region_trace_level >= level) {       \
	        printf args;                            \
	}                                               \
	MACRO_END
#define SHARED_REGION_TRACE_NONE(args)
#define SHARED_REGION_TRACE_ERROR(args)                         \
	MACRO_BEGIN                                             \
	SHARED_REGION_TRACE(SHARED_REGION_TRACE_ERROR_LVL,      \
	                    args);                              \
	MACRO_END
#define SHARED_REGION_TRACE_INFO(args)                          \
	MACRO_BEGIN                                             \
	SHARED_REGION_TRACE(SHARED_REGION_TRACE_INFO_LVL,       \
	                    args);                              \
	MACRO_END
#define SHARED_REGION_TRACE_DEBUG(args)                         \
	MACRO_BEGIN                                             \
	SHARED_REGION_TRACE(SHARED_REGION_TRACE_DEBUG_LVL,      \
	                    args);                              \
	MACRO_END

typedef struct vm_shared_region *vm_shared_region_t;

#ifndef MACH_KERNEL_PRIVATE
struct vm_shared_region;
struct vm_shared_region_slide_info;
struct vm_shared_region_slide_info_entry;
struct slide_info_entry_toc;
#endif /* MACH_KERNEL_PRIVATE */

struct _sr_file_mappings {
	int                     fd;
	uint32_t                mappings_count;
	struct shared_file_mapping_slide_np *mappings;
	uint32_t                slide;
	struct fileproc         *fp;
	struct vnode            *vp;
	memory_object_size_t    file_size;
	memory_object_control_t file_control;
};


#endif /* KERNEL_PRIVATE */

#endif  /* _VM_SHARED_REGION_H_ */
