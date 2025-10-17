/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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
#ifndef _BSD_KERN_MACH_FAT_H_
#define _BSD_KERN_MACH_FAT_H_

#include <mach/mach_types.h>
#include <kern/mach_loader.h>
#include <mach-o/fat.h>
#include <sys/vnode.h>

load_return_t fatfile_validate_fatarches(vm_offset_t data_ptr, vm_size_t data_size, off_t file_size);

load_return_t fatfile_getbestarch(vm_offset_t data_ptr, vm_size_t data_size, struct image_params *imgp, struct fat_arch *archret, bool affinity);
load_return_t fatfile_getbestarch_for_cputype(cpu_type_t cputype, cpu_subtype_t cpusubtype,
    vm_offset_t data_ptr, vm_size_t data_size, struct image_params *imgp, struct fat_arch *archret);
load_return_t fatfile_getarch_with_bits(integer_t archbits,
    vm_offset_t data_ptr, vm_size_t data_size, struct fat_arch *archret);

#endif /* _BSD_KERN_MACH_FAT_H_ */
