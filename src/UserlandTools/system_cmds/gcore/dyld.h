/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#include "options.h"
#include "corefile.h"
#include "utils.h"

#include <mach-o/dyld_images.h>
#include <mach-o/dyld_process_info.h>
#include <uuid/uuid.h>

#ifndef _DYLD_H
#define _DYLD_H

struct libent {
    const char *le_filename;            // (points into le_pathname!)
    char *le_pathname;
    uuid_t le_uuid;
    uint64_t le_mhaddr;                 // address in target process
    const native_mach_header_t *le_mh;  // copy mapped into this address space
	struct vm_range le_vr;				// vmaddr, vmsize bounds in target process
	mach_vm_offset_t le_objoff;			// offset from le_mhaddr to first __TEXT seg
};

extern const struct libent *libent_lookup_byuuid(const uuid_t);
extern const struct libent *libent_lookup_first_bytype(uint32_t);
extern const struct libent *libent_insert(const char *, const uuid_t, uint64_t, const native_mach_header_t *, const struct vm_range *, mach_vm_offset_t);
extern bool libent_build_nametable(task_t, dyld_process_info);

extern dyld_process_info get_task_dyld_info(task_t);
extern bool get_sc_uuid(dyld_process_info, uuid_t);
extern void free_task_dyld_info(dyld_process_info);
extern void create_dyld_header_regions(task_t task,struct regionhead *rhead);
extern void add_forced_regions(task_t task,struct regionhead *rhead);
extern bool is_range_part_of_the_shared_library_address_space(mach_vm_address_t address,mach_vm_size_t size);
#endif /* _DYLD_H */
