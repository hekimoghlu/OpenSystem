/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 14, 2025.
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
#include <kern/machine.h>
#include <kern/misc_protos.h>
#include <kern/thread.h>
#include <kern/processor.h>
#include <mach/machine.h>
#include <mach/processor_info.h>
#include <mach/mach_types.h>
#include <IOKit/IOPlatformExpert.h>

#include <IOKit/IOHibernatePrivate.h>
#include <vm/vm_page.h>
#include <vm/vm_pageout_xnu.h>
#include <vm/vm_purgeable_internal.h>
#include <vm/vm_compressor_xnu.h>
#include <kern/ecc.h>

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

boolean_t       need_to_unlock_decompressor = FALSE;

kern_return_t
hibernate_alloc_page_lists(
	hibernate_page_list_t ** page_list_ret,
	hibernate_page_list_t ** page_list_wired_ret,
	hibernate_page_list_t ** page_list_pal_ret)
{
	kern_return_t       retval = KERN_SUCCESS;

	hibernate_page_list_t * page_list = NULL;
	hibernate_page_list_t * page_list_wired = NULL;
	hibernate_page_list_t * page_list_pal = NULL;

	page_list = hibernate_page_list_allocate(TRUE);
	if (!page_list) {
		HIBLOG("%s: failed for page_list\n", __FUNCTION__);
		retval = KERN_RESOURCE_SHORTAGE;
		goto done;
	}
	page_list_wired = hibernate_page_list_allocate(FALSE);
	if (!page_list_wired) {
		kfree_data(page_list, page_list->list_size);
		HIBLOG("%s: failed for page_list_wired\n", __FUNCTION__);
		retval = KERN_RESOURCE_SHORTAGE;
		goto done;
	}
	page_list_pal = hibernate_page_list_allocate(FALSE);
	if (!page_list_pal) {
		kfree_data(page_list, page_list->list_size);
		kfree_data(page_list_wired, page_list_wired->list_size);
		HIBLOG("%s: failed for page_list_pal\n", __FUNCTION__);
		retval = KERN_RESOURCE_SHORTAGE;
		goto done;
	}
	*page_list_ret        = page_list;
	*page_list_wired_ret  = page_list_wired;
	*page_list_pal_ret    = page_list_pal;

done:
	return retval;
}

extern int sync_internal(void);

kern_return_t
hibernate_setup(IOHibernateImageHeader * header,
    boolean_t vmflush,
    hibernate_page_list_t * page_list __unused,
    hibernate_page_list_t * page_list_wired __unused,
    hibernate_page_list_t * page_list_pal __unused)
{
	kern_return_t       retval = KERN_SUCCESS;

	hibernate_create_paddr_map();

	hibernate_reset_stats();

	if (vmflush && VM_CONFIG_COMPRESSOR_IS_PRESENT) {
		sync_internal();

		vm_decompressor_lock();
		need_to_unlock_decompressor = TRUE;

		hibernate_flush_memory();
	}

	// no failures hereafter

	hibernate_processor_setup(header);

	HIBLOG("hibernate_alloc_pages act %d, inact %d, anon %d, throt %d, spec %d, wire %d, wireinit %d\n",
	    vm_page_active_count, vm_page_inactive_count,
	    vm_page_anonymous_count, vm_page_throttled_count, vm_page_speculative_count,
	    vm_page_wire_count, vm_page_wire_count_initial);

	if (retval != KERN_SUCCESS && need_to_unlock_decompressor == TRUE) {
		need_to_unlock_decompressor = FALSE;
		vm_decompressor_unlock();
	}
	return retval;
}

kern_return_t
hibernate_teardown(hibernate_page_list_t * page_list,
    hibernate_page_list_t * page_list_wired,
    hibernate_page_list_t * page_list_pal)
{
	hibernate_free_gobble_pages();

	if (page_list) {
		kfree_data(page_list, page_list->list_size);
	}
	if (page_list_wired) {
		kfree_data(page_list_wired, page_list_wired->list_size);
	}
	if (page_list_pal) {
		kfree_data(page_list_pal, page_list_pal->list_size);
	}

	if (VM_CONFIG_COMPRESSOR_IS_PRESENT) {
		if (need_to_unlock_decompressor == TRUE) {
			need_to_unlock_decompressor = FALSE;
			vm_decompressor_unlock();
		}
		vm_compressor_delay_trim();
	}


	return KERN_SUCCESS;
}
