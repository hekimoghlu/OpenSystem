/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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
#ifndef _VM_MEMORY_OBJECT_INTERNAL_H_
#define _VM_MEMORY_OBJECT_INTERNAL_H_

#ifdef XNU_KERNEL_PRIVATE

#include <vm/memory_object_xnu.h>

extern memory_object_default_t memory_manager_default;

__private_extern__
memory_object_default_t memory_manager_default_reference(void);

__private_extern__
kern_return_t           memory_manager_default_check(void);

__private_extern__
memory_object_control_t memory_object_control_allocate(
	vm_object_t             object);

__private_extern__
void                    memory_object_control_collapse(
	memory_object_control_t *control,
	vm_object_t             object);

__private_extern__
vm_object_t             memory_object_control_to_vm_object(
	memory_object_control_t control);
__private_extern__
vm_object_t             memory_object_to_vm_object(
	memory_object_t mem_obj);

extern void memory_object_control_disable(
	memory_object_control_t *control);

extern boolean_t        memory_object_is_shared_cache(
	memory_object_control_t         control);

#endif /* XNU_KERNEL_PRIVATE */

#endif  /* _VM_MEMORY_OBJECT_INTERNAL_H_ */
