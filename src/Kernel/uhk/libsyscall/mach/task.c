/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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
#undef _task_user_
#include <TargetConditionals.h>
#include <stdbool.h>

#include <mach/kern_return.h>
#include <mach/mach_param.h>
#include <mach/mach_port.h>
#include <mach/message.h>
#include <mach/mig_errors.h>
#include <mach/task_internal.h>
#include <mach/vm_map.h>

extern mach_port_t      mach_task_self_;

boolean_t
mach_task_is_self(task_name_t task)
{
	boolean_t is_self;
	kern_return_t kr;

	if (task == mach_task_self_) {
		return TRUE;
	}

	kr = _kernelrpc_mach_task_is_self(task, &is_self);

	return kr == KERN_SUCCESS && is_self;
}


kern_return_t
mach_ports_register(
	task_t                  target_task,
	mach_port_array_t       init_port_set,
	mach_msg_type_number_t  init_port_setCnt)
{
	mach_port_t array[TASK_PORT_REGISTER_MAX] = { };
	kern_return_t kr;

	if (init_port_setCnt > TASK_PORT_REGISTER_MAX) {
		return KERN_INVALID_ARGUMENT;
	}

	for (mach_msg_type_number_t i = 0; i < init_port_setCnt; i++) {
		array[i] = init_port_set[i];
	}

	kr = _kernelrpc_mach_ports_register3(target_task, array[0], array[1], array[2]);
	return kr;
}

kern_return_t
mach_ports_lookup(
	task_t                  target_task,
	mach_port_array_t      *init_port_set,
	mach_msg_type_number_t *init_port_setCnt)
{
	vm_size_t size = TASK_PORT_REGISTER_MAX * sizeof(mach_port_t);
	mach_port_array_t array;
	vm_address_t addr = 0;
	kern_return_t kr;

	kr = vm_allocate(target_task, &addr, size, VM_FLAGS_ANYWHERE);
	array = (mach_port_array_t)addr;
	if (kr != KERN_SUCCESS) {
		return kr;
	}


	kr = _kernelrpc_mach_ports_lookup3(target_task,
	    &array[0], &array[1], &array[2]);
	if (kr != KERN_SUCCESS) {
		vm_deallocate(target_task, addr, size);
		return kr;
	}

	*init_port_set = array;
	*init_port_setCnt = TASK_PORT_REGISTER_MAX;
	return KERN_SUCCESS;
}
