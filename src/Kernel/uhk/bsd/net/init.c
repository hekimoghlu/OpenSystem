/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 28, 2023.
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
#include <kern/kalloc.h>
#include <libkern/OSAtomic.h>
#include <sys/errno.h>
#include <sys/sysctl.h>
#include <net/init.h>
#include <libkern/libkern.h>
#include <os/atomic_private.h>
#include <pexpert/pexpert.h>
#include <string.h>

#if (DEBUG | DEVELOPMENT)
SYSCTL_DECL(_net_diagnose);
SYSCTL_NODE(_net, OID_AUTO, diagnose, CTLFLAG_RW | CTLFLAG_LOCKED, 0, "");

static int net_diagnose_on = 0;
SYSCTL_INT(_net_diagnose, OID_AUTO, on,
    CTLFLAG_RW | CTLFLAG_LOCKED, &net_diagnose_on, 0, "");
#endif /* (DEBUG | DEVELOPMENT) */

struct init_list_entry {
	struct init_list_entry  *next;
	net_init_func_ptr               func;
};

static struct init_list_entry *LIST_RAN = __unsafe_forge_single(struct init_list_entry *, ~(uintptr_t)0);
static struct init_list_entry *list_head = NULL;

errno_t
net_init_add(net_init_func_ptr init_func)
{
	struct init_list_entry  *entry;

	if (init_func == 0) {
		return EINVAL;
	}

	/* Check if we've already started */
	if (list_head == LIST_RAN) {
		return EALREADY;
	}

	entry = kalloc_type(struct init_list_entry,
	    Z_WAITOK | Z_ZERO | Z_NOFAIL);

	entry->func = init_func;

	do {
		entry->next = list_head;

		if (entry->next == LIST_RAN) {
			/* List already ran, cleanup and call the function */
			kfree_type(struct init_list_entry, entry);
			return EALREADY;
		}
	} while (!os_atomic_cmpxchg(&list_head, entry->next, entry, acq_rel));

	return 0;
}

__private_extern__ void
net_init_run(void)
{
	struct init_list_entry  *__single backward_head = NULL;
	struct init_list_entry  *__single forward_head = NULL;
	struct init_list_entry  *__single current = NULL;

	/*
	 * Grab the list, replacing the head with 0xffffffff to indicate
	 * that we've already run.
	 */
	do {
		backward_head = list_head;
	} while (!os_atomic_cmpxchg(&list_head, backward_head, LIST_RAN, acq_rel));

	/* Reverse the order of the list */
	while (backward_head != 0) {
		current = backward_head;
		backward_head = current->next;
		current->next = forward_head;
		forward_head = current;
	}

	/* Call each function pointer registered */
	while (forward_head != 0) {
		current = forward_head;
		forward_head = current->next;
		current->func();
		kfree_type(struct init_list_entry, current);
	}

#if (DEBUG || DEVELOPMENT)
	(void) PE_parse_boot_argn("net_diagnose_on", &net_diagnose_on, sizeof(net_diagnose_on));
#endif /* (DEBUG || DEVELOPMENT) */
}
