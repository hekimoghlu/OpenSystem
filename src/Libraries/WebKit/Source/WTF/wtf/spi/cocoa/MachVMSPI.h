/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 19, 2022.
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
#pragma once

#include <mach/boolean.h>
#include <mach/kern_return.h>
#include <mach/mach_types.h>

#if PLATFORM(MAC) || USE(APPLE_INTERNAL_SDK)
#include <mach/mach_vm.h>
#endif

WTF_EXTERN_C_BEGIN

kern_return_t mach_vm_allocate(vm_map_t target, mach_vm_address_t*, mach_vm_size_t, int flags);
kern_return_t mach_vm_deallocate(vm_map_t target, mach_vm_address_t, mach_vm_size_t);
kern_return_t mach_vm_map(vm_map_t targetTask, mach_vm_address_t*, mach_vm_size_t, mach_vm_offset_t mask, int flags,
    mem_entry_name_port_t, memory_object_offset_t, boolean_t copy, vm_prot_t currentProtection, vm_prot_t maximumProtection, vm_inherit_t);
kern_return_t mach_vm_remap(vm_map_t targetTask, mach_vm_address_t*, mach_vm_size_t, mach_vm_offset_t mask, int flags,
    vm_map_t srcTask, mach_vm_address_t srcAddress, boolean_t copy, vm_prot_t* currentProtection, vm_prot_t* maximumProtection, vm_inherit_t);
kern_return_t mach_vm_protect(vm_map_t targetTask, mach_vm_address_t, mach_vm_size_t, boolean_t setMaximum, vm_prot_t newProtection);
kern_return_t mach_vm_region(vm_map_t targetTask, mach_vm_address_t*, mach_vm_size_t*, vm_region_flavor_t, vm_region_info_t,
    mach_msg_type_number_t* infoCount, mach_port_t* objectName);
kern_return_t mach_vm_region_recurse(vm_map_t targetTask, mach_vm_address_t*, mach_vm_size_t*, uint32_t* depth, vm_region_recurse_info_t, mach_msg_type_number_t* infoCount);
kern_return_t mach_vm_purgable_control(vm_map_t target, mach_vm_address_t, vm_purgable_t control, int* state);

WTF_EXTERN_C_END
