/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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
#include <voucher/ipc_pthread_priority_types.h>
#include <mach/mach_types.h>
#include <mach/kern_return.h>
#include <ipc/ipc_port.h>
#include <mach/mach_vm.h>
#include <mach/vm_map.h>
#include <vm/vm_map.h>
#include <mach/host_priv.h>
#include <mach/host_special_ports.h>
#include <kern/host.h>
#include <kern/ledger.h>
#include <sys/kdebug.h>
#include <IOKit/IOBSD.h>
#include <pthread/priority_private.h>

ipc_voucher_attr_control_t  ipc_pthread_priority_voucher_attr_control;    /* communication channel from PTHPRIORITY to voucher system */

#define PTHPRIORITY_ATTR_DEFAULT_VALUE (0)

#define IPC_PTHREAD_PRIORITY_VALUE_TO_HANDLE(x) ((mach_voucher_attr_value_handle_t)(x))
#define HANDLE_TO_IPC_PTHREAD_PRIORITY_VALUE(x) ((ipc_pthread_priority_value_t)(x))

kern_return_t
ipc_pthread_priority_release_value(
	ipc_voucher_attr_manager_t __assert_only manager,
	mach_voucher_attr_key_t __assert_only key,
	mach_voucher_attr_value_handle_t value,
	mach_voucher_attr_value_reference_t sync);

kern_return_t
ipc_pthread_priority_get_value(
	ipc_voucher_attr_manager_t __assert_only manager,
	mach_voucher_attr_key_t __assert_only key,
	mach_voucher_attr_recipe_command_t command,
	mach_voucher_attr_value_handle_array_t prev_values,
	mach_msg_type_number_t __assert_only prev_value_count,
	mach_voucher_attr_content_t recipe,
	mach_voucher_attr_content_size_t recipe_size,
	mach_voucher_attr_value_handle_t *out_value,
	mach_voucher_attr_value_flags_t  *out_flags,
	ipc_voucher_t *out_value_voucher);

kern_return_t
ipc_pthread_priority_extract_content(
	ipc_voucher_attr_manager_t __assert_only manager,
	mach_voucher_attr_key_t __assert_only key,
	mach_voucher_attr_value_handle_array_t values,
	mach_msg_type_number_t value_count,
	mach_voucher_attr_recipe_command_t *out_command,
	mach_voucher_attr_content_t out_recipe,
	mach_voucher_attr_content_size_t *in_out_recipe_size);

kern_return_t
ipc_pthread_priority_command(
	ipc_voucher_attr_manager_t __assert_only manager,
	mach_voucher_attr_key_t __assert_only key,
	mach_voucher_attr_value_handle_array_t values,
	mach_msg_type_number_t value_count,
	mach_voucher_attr_command_t command,
	mach_voucher_attr_content_t in_content,
	mach_voucher_attr_content_size_t in_content_size,
	mach_voucher_attr_content_t out_content,
	mach_voucher_attr_content_size_t *in_out_content_size);

/*
 * communication channel from voucher system to IPC_PTHREAD_PRIORITY
 */
const struct ipc_voucher_attr_manager ipc_pthread_priority_manager = {
	.ivam_release_value    = ipc_pthread_priority_release_value,
	.ivam_get_value        = ipc_pthread_priority_get_value,
	.ivam_extract_content  = ipc_pthread_priority_extract_content,
	.ivam_command          = ipc_pthread_priority_command,
	.ivam_flags            = IVAM_FLAGS_NONE,
};

/*
 * Routine: ipc_pthread_priority_init
 * Purpose: Initialize the IPC_PTHREAD_PRIORITY subsystem.
 * Returns: None.
 */
__startup_func
static void
ipc_pthread_priority_init(void)
{
	/* Register the ipc_pthread_priority manager with the Vouchers sub system. */
	ipc_register_well_known_mach_voucher_attr_manager(
		&ipc_pthread_priority_manager,
		0,
		MACH_VOUCHER_ATTR_KEY_PTHPRIORITY,
		&ipc_pthread_priority_voucher_attr_control);

	kprintf("IPC_PTHREAD_PRIORITY subsystem is initialized\n");
}
STARTUP(MACH_IPC, STARTUP_RANK_FIRST, ipc_pthread_priority_init);

/*
 * IPC_PTHREAD_PRIORITY Resource Manager Routines.
 */


/*
 * Routine: ipc_pthread_priority_release_value
 * Purpose: Release a value, if sync matches the sync count in value.
 * Returns: KERN_SUCCESS: on Successful deletion.
 *          KERN_FAILURE: if sync value does not matches.
 */
kern_return_t
ipc_pthread_priority_release_value(
	ipc_voucher_attr_manager_t              __assert_only manager,
	mach_voucher_attr_key_t                 __assert_only key,
	mach_voucher_attr_value_handle_t                      value,
	mach_voucher_attr_value_reference_t                   sync)
{
	assert(MACH_VOUCHER_ATTR_KEY_PTHPRIORITY == key);
	assert(manager == &ipc_pthread_priority_manager);

	ipc_pthread_priority_value_t ipc_pthread_priority_value = HANDLE_TO_IPC_PTHREAD_PRIORITY_VALUE(value);

	panic("ipc_pthread_priority_release_value called for a persistent PTHPRIORITY value %x with sync value %d", ipc_pthread_priority_value, sync);
	return KERN_FAILURE;
}

/*
 * Routine: ipc_pthread_priority_get_value
 */
kern_return_t
ipc_pthread_priority_get_value(
	ipc_voucher_attr_manager_t              __assert_only manager,
	mach_voucher_attr_key_t                 __assert_only key,
	mach_voucher_attr_recipe_command_t                command,
	mach_voucher_attr_value_handle_array_t __unused   prev_values,
	mach_msg_type_number_t                 __unused   prev_value_count,
	mach_voucher_attr_content_t                       recipe,
	mach_voucher_attr_content_size_t                  recipe_size,
	mach_voucher_attr_value_handle_t             *out_value,
	mach_voucher_attr_value_flags_t              *out_flags,
	ipc_voucher_t                                            *out_value_voucher)
{
	kern_return_t kr = KERN_SUCCESS;
	ipc_pthread_priority_value_t ipc_pthread_priority_value;
	ipc_pthread_priority_value_t canonicalize_priority_value;

	assert(MACH_VOUCHER_ATTR_KEY_PTHPRIORITY == key);
	assert(manager == &ipc_pthread_priority_manager);

	/* never an out voucher */
	*out_value_voucher = IPC_VOUCHER_NULL;
	*out_flags = MACH_VOUCHER_ATTR_VALUE_FLAGS_NONE;

	switch (command) {
	case MACH_VOUCHER_ATTR_PTHPRIORITY_CREATE:

		if (recipe_size != sizeof(ipc_pthread_priority_value_t)) {
			return KERN_INVALID_ARGUMENT;
		}

		memcpy(&ipc_pthread_priority_value, recipe, recipe_size);

		if (ipc_pthread_priority_value == PTHPRIORITY_ATTR_DEFAULT_VALUE) {
			*out_value = IPC_PTHREAD_PRIORITY_VALUE_TO_HANDLE(PTHPRIORITY_ATTR_DEFAULT_VALUE);
			return kr;
		}

		/* Callout to pthread kext to get the canonicalized value */
		canonicalize_priority_value = (ipc_pthread_priority_value_t)
		    _pthread_priority_normalize_for_ipc((unsigned long)ipc_pthread_priority_value);

		*out_value = IPC_PTHREAD_PRIORITY_VALUE_TO_HANDLE(canonicalize_priority_value);
		*out_flags = MACH_VOUCHER_ATTR_VALUE_FLAGS_PERSIST;
		return kr;

	default:
		kr = KERN_INVALID_ARGUMENT;
		break;
	}

	return kr;
}

/*
 * Routine: ipc_pthread_priority_extract_content
 * Purpose: Extract a set of pthread_priority value from an array of voucher values.
 * Returns: KERN_SUCCESS: on Success.
 *          KERN_NO_SPACE: insufficeint buffer provided to fill an array of pthread_priority values.
 */
kern_return_t
ipc_pthread_priority_extract_content(
	ipc_voucher_attr_manager_t      __assert_only manager,
	mach_voucher_attr_key_t         __assert_only key,
	mach_voucher_attr_value_handle_array_t        values,
	mach_msg_type_number_t                                value_count,
	mach_voucher_attr_recipe_command_t           *out_command,
	mach_voucher_attr_content_t                   out_recipe,
	mach_voucher_attr_content_size_t             *in_out_recipe_size)
{
	kern_return_t kr = KERN_SUCCESS;
	mach_msg_type_number_t i;
	ipc_pthread_priority_value_t ipc_pthread_priority_value;

	assert(MACH_VOUCHER_ATTR_KEY_PTHPRIORITY == key);
	assert(manager == &ipc_pthread_priority_manager);

	for (i = 0; i < value_count && *in_out_recipe_size > 0; i++) {
		ipc_pthread_priority_value = HANDLE_TO_IPC_PTHREAD_PRIORITY_VALUE(values[i]);

		if (ipc_pthread_priority_value == PTHPRIORITY_ATTR_DEFAULT_VALUE) {
			continue;
		}

		if (MACH_VOUCHER_PTHPRIORITY_CONTENT_SIZE > *in_out_recipe_size) {
			*in_out_recipe_size = 0;
			return KERN_NO_SPACE;
		}

		memcpy(&out_recipe[0], &ipc_pthread_priority_value, sizeof(ipc_pthread_priority_value));
		*out_command = MACH_VOUCHER_ATTR_PTHPRIORITY_NULL;
		*in_out_recipe_size = (mach_voucher_attr_content_size_t)sizeof(ipc_pthread_priority_value);
		return kr;
	}

	*in_out_recipe_size = 0;
	return KERN_INVALID_VALUE;
}

/*
 * Routine: ipc_pthread_priority_command
 * Purpose: Execute a command against a set of PTHPRIORITY values.
 * Returns: KERN_SUCCESS: On successful execution of command.
 *          KERN_FAILURE: On failure.
 */
kern_return_t
ipc_pthread_priority_command(
	ipc_voucher_attr_manager_t                 __assert_only manager,
	mach_voucher_attr_key_t                    __assert_only key,
	mach_voucher_attr_value_handle_array_t  __unused values,
	mach_msg_type_number_t                  __unused value_count,
	mach_voucher_attr_command_t              __unused command,
	mach_voucher_attr_content_t        __unused in_content,
	mach_voucher_attr_content_size_t   __unused in_content_size,
	mach_voucher_attr_content_t        __unused out_content,
	mach_voucher_attr_content_size_t   __unused *out_content_size)
{
	assert(MACH_VOUCHER_ATTR_KEY_PTHPRIORITY == key);
	assert(manager == &ipc_pthread_priority_manager);

	return KERN_FAILURE;
}
