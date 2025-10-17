/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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
#include <vm/pmap.h>
#include <vm/vm_page_internal.h>

/**
 * Initialize a unified page list iterator object from a source list of pages.
 * The created iterator will point to the beginning of the list.
 *
 * @note Where applicable, we expect the calling VM code to hold locks that prevent
 *       the underlying page lists from being concurrently changed from underneath
 *       the page list and page list iterator.  In particular, for
 *       UNIFIED_PAGE_LIST_TYPE_VM_PAGE_OBJ_Q this means holding the VM object lock,
 *       and for UNIFIED_PAGE_LIST_TYPE_VM_PAGE_FIFO_Q this means holding any
 *       relevant global page queue lock.
 *
 * @param page_list The list of pages to serve as the starting point of the iterator.
 * @param iter Output parameter to store the initialized iterator.
 */
__attribute__((always_inline))
void
unified_page_list_iterator_init(
	const unified_page_list_t *page_list,
	unified_page_list_iterator_t *iter)
{
	switch (page_list->type) {
	case UNIFIED_PAGE_LIST_TYPE_UPL_ARRAY:
		iter->upl_index = 0;
		break;
	case UNIFIED_PAGE_LIST_TYPE_VM_PAGE_LIST:
		iter->pageq_pos = page_list->page_slist;
		break;
	case UNIFIED_PAGE_LIST_TYPE_VM_PAGE_OBJ_Q:
	case UNIFIED_PAGE_LIST_TYPE_VM_PAGE_FIFO_Q:
		iter->pageq_pos = (vm_page_t)vm_page_queue_first((vm_page_queue_head_t*)page_list->pageq);
		break;
	default:
		panic("%s: Unrecognized page list type %hhu", __func__, page_list->type);
	}
	iter->list = page_list;
}

/**
 * Move to the next element within a page list iterator.
 *
 * @note unified_page_list_iterator_end() should be used to avoid iterating
 *       past the end of the list.
 *
 * @param iter The iterator to advance.
 */
__attribute__((always_inline))
void
unified_page_list_iterator_next(unified_page_list_iterator_t *iter)
{
	switch (iter->list->type) {
	case UNIFIED_PAGE_LIST_TYPE_UPL_ARRAY:
		iter->upl_index++;
		break;
	case UNIFIED_PAGE_LIST_TYPE_VM_PAGE_LIST:
		iter->pageq_pos = NEXT_PAGE(iter->pageq_pos);
		break;
	case UNIFIED_PAGE_LIST_TYPE_VM_PAGE_OBJ_Q:
		iter->pageq_pos = (vm_page_t)vm_page_queue_next(&iter->pageq_pos->vmp_listq);
		break;
	case UNIFIED_PAGE_LIST_TYPE_VM_PAGE_FIFO_Q:
		iter->pageq_pos = (vm_page_t)vm_page_queue_next(&iter->pageq_pos->vmp_pageq);
		break;
	}
}

/**
 * Determine whether the current position of a page list iterator is at the
 * end of the list.
 *
 * @note The behavior of this function is undefined if the caller has already advanced
 *       the iterator beyond the end of the list.
 *
 * @param iter The iterator to check.
 *
 * @return True if the iterator has reached the end of the list, false otherwise.
 */
__attribute__((always_inline))
bool
unified_page_list_iterator_end(const unified_page_list_iterator_t *iter)
{
	switch (iter->list->type) {
	case UNIFIED_PAGE_LIST_TYPE_UPL_ARRAY:
		return iter->upl_index >= iter->list->upl.upl_size;
	case UNIFIED_PAGE_LIST_TYPE_VM_PAGE_LIST:
		return iter->pageq_pos == NULL;
	case UNIFIED_PAGE_LIST_TYPE_VM_PAGE_OBJ_Q:
	case UNIFIED_PAGE_LIST_TYPE_VM_PAGE_FIFO_Q:
		return vm_page_queue_end((vm_page_queue_head_t*)iter->list->pageq, (vm_page_queue_entry_t)iter->pageq_pos);
	}
}

/**
 * Extract the physical page number from the current position of a page list iter.
 *
 * @note The behavior of this function is undefined if the iterator is already at or
 *       beyond the end of the page list.
 *
 * @param iter The iterator from which to extract the current page.
 * @param is_fictitious Output parameter indicating whether the current iterator position
 *        represents a fictitious page.  Useful for pmap functions that are meant to only
 *        operate on real physical pages.
 *
 * @return The physical page number of the current iterator position.
 */
__attribute__((always_inline))
ppnum_t
unified_page_list_iterator_page(
	const unified_page_list_iterator_t *iter,
	bool *is_fictitious)
{
	ppnum_t phys_page;
	switch (iter->list->type) {
	case UNIFIED_PAGE_LIST_TYPE_UPL_ARRAY:
		phys_page = iter->list->upl.upl_info[iter->upl_index].phys_addr;
		*is_fictitious = false;
		break;
	case UNIFIED_PAGE_LIST_TYPE_VM_PAGE_LIST:
	case UNIFIED_PAGE_LIST_TYPE_VM_PAGE_OBJ_Q:
	case UNIFIED_PAGE_LIST_TYPE_VM_PAGE_FIFO_Q:
		phys_page = VM_PAGE_GET_PHYS_PAGE(iter->pageq_pos);
		*is_fictitious = (vm_page_is_fictitious(iter->pageq_pos) != 0);
		break;
	}

	return phys_page;
}
