/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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
#ifndef _VM_VM_PAGEOUT_INTERNAL_H_
#define _VM_VM_PAGEOUT_INTERNAL_H_

#include <sys/cdefs.h>
#include <vm/vm_pageout_xnu.h>

__BEGIN_DECLS

#ifdef XNU_KERNEL_PRIVATE

#ifdef MACH_KERNEL_PRIVATE

#define VM_PAGEOUT_GC_INIT      ((void *)0)
#define VM_PAGEOUT_GC_COLLECT   ((void *)1)
extern void vm_pageout_garbage_collect(void *, wait_result_t);

/* UPL exported routines and structures */

#define upl_lock_init(object)   lck_mtx_init(&(object)->Lock, &vm_object_lck_grp, &vm_object_lck_attr)
#define upl_lock_destroy(object)        lck_mtx_destroy(&(object)->Lock, &vm_object_lck_grp)
#define upl_lock(object)        lck_mtx_lock(&(object)->Lock)
#define upl_unlock(object)      lck_mtx_unlock(&(object)->Lock)
#define upl_try_lock(object)    lck_mtx_try_lock(&(object)->Lock)
#define upl_lock_sleep(object, event, thread)                           \
	lck_mtx_sleep_with_inheritor(&(object)->Lock,                   \
	              LCK_SLEEP_DEFAULT,                                \
	              (event_t) (event),                                \
	              (thread),                                         \
	              THREAD_UNINT,                                     \
	              TIMEOUT_WAIT_FOREVER)
#define upl_wakeup(event) wakeup_all_with_inheritor((event), THREAD_AWAKENED)

extern void vm_object_set_pmap_cache_attr(
	vm_object_t             object,
	upl_page_info_array_t   user_page_list,
	unsigned int            num_pages,
	boolean_t               batch_pmap_op);

extern kern_return_t vm_object_iopl_request(
	vm_object_t             object,
	vm_object_offset_t      offset,
	upl_size_t              size,
	upl_t                  *upl_ptr,
	upl_page_info_array_t   user_page_list,
	unsigned int           *page_list_count,
	upl_control_flags_t     cntrl_flags,
	vm_tag_t                tag);

/* should be just a regular vm_map_enter() */
extern kern_return_t vm_map_enter_upl(
	vm_map_t                map,
	upl_t                   upl,
	vm_map_offset_t         *dst_addr);

/* should be just a regular vm_map_remove() */
extern kern_return_t vm_map_remove_upl(
	vm_map_t                map,
	upl_t                   upl);

extern kern_return_t vm_map_enter_upl_range(
	vm_map_t                map,
	upl_t                   upl,
	vm_object_offset_t             offset,
	vm_size_t               size,
	vm_prot_t               prot,
	vm_map_offset_t         *dst_addr);

extern kern_return_t vm_map_remove_upl_range(
	vm_map_t                map,
	upl_t                   upl,
	vm_object_offset_t             offset,
	vm_size_t               size);


extern struct vm_page_delayed_work*
vm_page_delayed_work_get_ctx(void);

extern void
vm_page_delayed_work_finish_ctx(struct vm_page_delayed_work* dwp);

extern void vm_pageout_throttle_up(vm_page_t page);

extern kern_return_t vm_paging_map_object(
	vm_page_t               page,
	vm_object_t             object,
	vm_object_offset_t      offset,
	vm_prot_t               protection,
	boolean_t               can_unlock_object,
	vm_map_size_t           *size,          /* IN/OUT */
	vm_map_offset_t         *address,       /* OUT */
	boolean_t               *need_unmap);   /* OUT */
extern void vm_paging_unmap_object(
	vm_object_t             object,
	vm_map_offset_t         start,
	vm_map_offset_t         end);
decl_simple_lock_data(extern, vm_paging_lock);


/*
 * Backing store throttle when BS is exhausted
 */
extern unsigned int    vm_backing_store_low;

extern void vm_pageout_steal_laundry(
	vm_page_t page,
	boolean_t queues_locked);


#endif /* MACH_KERNEL_PRIVATE */

#endif /* XNU_KERNEL_PRIVATE */
__END_DECLS

#endif  /* _VM_VM_PAGEOUT_INTERNAL_H_ */
