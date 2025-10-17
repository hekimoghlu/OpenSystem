/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 17, 2023.
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
// These declarations must match those in Libc's stack_logging.h

#include <mach/vm_statistics.h>         // to get VM_FLAGS_ALIAS_MASK

#define stack_logging_type_vm_allocate 16      // mach_vm_allocate, mmap, mach_vm_map, mach_vm_remap, etc
#define stack_logging_type_vm_deallocate 32    // mach_vm_deallocate or munmap
#define stack_logging_type_mapped_file_or_shared_mem 128    // a hint that the VM region *might* be a mapped file or shared memory

// For logging VM allocation and deallocation, arg1 here
// is the mach_port_name_t of the target task in which the
// alloc or dealloc is occurring. For example, for mmap()
// that would be mach_task_self(), but for a cross-task-capable
// call such as mach_vm_map(), it is the target task.

typedef void (malloc_logger_t)(uint32_t type, uintptr_t arg1, uintptr_t arg2, uintptr_t arg3, uintptr_t result, uint32_t num_hot_frames_to_skip);

extern malloc_logger_t *__syscall_logger;
