/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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
#include <mach/mach_types.h>
#include <mach/notify.h>
#include <ipc/ipc_port.h>
#include <kern/ipc_kobject.h>
#include <kern/ipc_misc.h>

#include <mach/mach_port.h>
#include <mach/vm_map.h>
#include <vm/vm_map.h>
#include <vm/vm_kern.h>

extern void fileport_releasefg(struct fileglob *);

/*
 * fileport_alloc
 *
 * Description: Obtain a send right for the given fileglob, which must be
 *		referenced.
 *
 * Parameters:  fg		A fileglob.
 *
 * Returns:     Port of type IKOT_FILEPORT with fileglob set as its kobject.
 *              Port is returned with a send right.
 */
ipc_port_t
fileport_alloc(struct fileglob *fg)
{
	return ipc_kobject_alloc_port((ipc_kobject_t)fg, IKOT_FILEPORT,
	           IPC_KOBJECT_ALLOC_MAKE_SEND | IPC_KOBJECT_ALLOC_NSREQUEST);
}


/*
 * fileport_get_fileglob
 *
 * Description: Obtain the fileglob associated with a given port.
 *
 * Parameters: port		A Mach port of type IKOT_FILEPORT.
 *
 * Returns:    NULL		The given Mach port did not reference a
 *				fileglob.
 *	       !NULL		The fileglob that is associated with the
 *				Mach port.
 *
 * Notes: The caller must have a reference on the fileport.
 */
struct fileglob *
fileport_port_to_fileglob(ipc_port_t port)
{
	if (IP_VALID(port)) {
		return ipc_kobject_get_stable(port, IKOT_FILEPORT);
	}
	return NULL;
}


/*
 * fileport_no_senders
 *
 * Description: Handle a no-senders notification for a fileport.  Unless
 *              the message is spoofed, destroys the port and releases
 *              its reference on the fileglob.
 *
 * Parameters: msg		A Mach no-senders notification message.
 */
static void
fileport_no_senders(ipc_port_t port, mach_port_mscount_t mscount)
{
	struct fileglob *fg;

	fg = ipc_kobject_dealloc_port(port, mscount, IKOT_FILEPORT);

	fileport_releasefg(fg);
}

IPC_KOBJECT_DEFINE(IKOT_FILEPORT,
    .iko_op_stable     = true,
    .iko_op_no_senders = fileport_no_senders);

/*
 * fileport_invoke
 *
 * Description: Invoke a function with the fileglob underlying the fileport.
 *		Returns the error code related to the fileglob lookup.
 *
 * Parameters:	task		The target task
 *		action		The function to invoke with the fileglob
 *		arg		Anonymous pointer to caller state
 *		rval		The value returned from calling 'action'
 */
kern_return_t
fileport_invoke(task_t task, mach_port_name_t name,
    int (*action)(mach_port_name_t, struct fileglob *, void *),
    void *arg, int *rval)
{
	kern_return_t kr;
	ipc_port_t fileport;
	struct fileglob *fg;

	kr = ipc_typed_port_copyin_send(task->itk_space, name,
	    IKOT_FILEPORT, &fileport);
	if (kr != KERN_SUCCESS) {
		return kr;
	}

	if ((fg = fileport_port_to_fileglob(fileport)) != NULL) {
		*rval = (*action)(name, fg, arg);
	} else {
		kr = KERN_FAILURE;
	}
	ipc_typed_port_release_send(fileport, IKOT_FILEPORT);
	return kr;
}

/*
 * fileport_walk
 *
 * Description: Invoke the action function on every fileport in the task.
 *
 * Parameters:  task		The target task
 *		countp		Returns how many ports were found
 *		action		The function to invoke on each fileport
 */
kern_return_t
fileport_walk(task_t task, size_t *countp,
    bool (^cb)(size_t i, mach_port_name_t, struct fileglob *))
{
	const uint32_t BATCH_SIZE = 4 << 10;
	ipc_space_t space = task->itk_space;
	ipc_entry_table_t table;
	ipc_entry_num_t index;
	ipc_entry_t entry;
	size_t count = 0;

	is_read_lock(space);
	if (!is_active(space)) {
		is_read_unlock(space);
		return KERN_INVALID_TASK;
	}

	table = is_active_table(space);
	entry = ipc_entry_table_base(table);

	/* skip the first element which is not a real entry */
	index = 1;
	entry = ipc_entry_table_next_elem(table, entry);

	for (;;) {
		ipc_entry_bits_t bits = entry->ie_bits;
		mach_port_name_t name;
		struct fileglob *fg;

		if (IE_BITS_TYPE(bits) & MACH_PORT_TYPE_SEND) {
			ipc_port_t port = entry->ie_port;

			name = MACH_PORT_MAKE(index, IE_BITS_GEN(bits));
			fg   = fileport_port_to_fileglob(port);

			if (fg) {
				if (cb && !cb(count, name, fg)) {
					cb = NULL;
					if (countp == NULL) {
						break;
					}
				}
				count++;
			}
		}

		index++;
		entry = ipc_entry_table_next_elem(table, entry);
		if (!entry) {
			break;
		}
		if (index % BATCH_SIZE == 0) {
			/*
			 * Give the system some breathing room,
			 * validate that the space is still valid,
			 * and reload the pointer and length.
			 */
			is_read_unlock(space);
			is_read_lock(space);
			if (!is_active(space)) {
				is_read_unlock(space);
				return KERN_INVALID_TASK;
			}

			table = is_active_table(space);
			entry = ipc_entry_table_get_nocheck(table, index);
		}
	}

	is_read_unlock(space);

	if (countp) {
		*countp = count;
	}

	return KERN_SUCCESS;
}
