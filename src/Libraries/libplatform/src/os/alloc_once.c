/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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
#include "os/internal.h"
#include "resolver.h"
#include "os/alloc_once_impl.h"
#include <mach/mach_init.h>
#include <mach/mach_vm.h>
#include <mach/vm_statistics.h>

#pragma mark -
#pragma mark os_alloc

typedef struct _os_alloc_heap_metadata_s {
	size_t allocated_bytes;
	void *prev;
} _os_alloc_heap_metadata_s;

#define allocation_size (2 * vm_page_size)
#define usable (allocation_size-sizeof(_os_alloc_heap_metadata_s))
OS_NOEXPORT void * volatile _os_alloc_heap;

OS_ATOMIC_EXPORT void* _os_alloc_once(struct _os_alloc_once_s *slot, size_t sz,
		os_function_t init);

void * volatile _os_alloc_heap;

/*
 * Simple allocator that doesn't have to worry about ever freeing allocations.
 *
 * The heapptr entry of _os_alloc_once_metadata always points to the newest
 * available heap page, or NULL if this is the first allocation. The heap has a
 * small header at the top of each heap block, recording the currently
 * allocated bytes and the pointer to the previous heap block.
 *
 * Ignoring the special case where the heapptr is NULL; in which case we always
 * make a block. The allocator first atomically increments the allocated_bytes
 * counter by sz and calculates the eventual base pointer. If base+sz is
 * greater than allocation_size then we begin allocating a new page. Otherwise,
 * base is returned.
 *
 * Page allocation vm_allocates a new page of allocation_size and then attempts
 * to atomically cmpxchg that pointer with the current headptr. If successful,
 * it links the previous page to the new heap block for debugging purposes and
 * then reattempts allocation. If a thread loses the allocation race, it
 * vm_deallocates the still-clean region and reattempts the whole allocation.
 */

static inline void*
_os_alloc_alloc(void *heap, size_t sz)
{
	if (likely(heap)) {
		_os_alloc_heap_metadata_s *metadata = (_os_alloc_heap_metadata_s*)heap;
		size_t used = os_atomic_add(&metadata->allocated_bytes, sz, relaxed);
		if (likely(used <= usable)) {
			return ((char*)metadata + sizeof(_os_alloc_heap_metadata_s) +
					used - sz);
		}
	}
	/* This fall-through case is heap == NULL, or heap block is exhausted. */
	return NULL;
}

OS_NOINLINE
static void*
_os_alloc_slow(void *heap, size_t sz)
{
	void *ptr;
	do {
		/*
		 * <rdar://problem/13208498> We allocate at PAGE_SIZE or above to ensure
		 * we don't land in the zero page *if* a binary has opted not to include
		 * the __PAGEZERO load command.
		 */
		mach_vm_address_t heapblk = PAGE_SIZE;
		kern_return_t kr;
		kr = mach_vm_map(mach_task_self(), &heapblk, allocation_size,
				0, VM_FLAGS_ANYWHERE | VM_MAKE_TAG(VM_MEMORY_OS_ALLOC_ONCE),
				MEMORY_OBJECT_NULL, 0, FALSE, VM_PROT_DEFAULT, VM_PROT_ALL,
				VM_INHERIT_DEFAULT);
		if (unlikely(kr)) {
			__LIBPLATFORM_INTERNAL_CRASH__(kr, "Failed to allocate in os_alloc_once");
		}
		if (os_atomic_cmpxchg(&_os_alloc_heap, heap, (void*)heapblk, relaxed)) {
			((_os_alloc_heap_metadata_s*)heapblk)->prev = heap;
			heap = (void*)heapblk;
		} else {
			mach_vm_deallocate(mach_task_self(), heapblk, allocation_size);
			heap = _os_alloc_heap;
		}
		ptr = _os_alloc_alloc(heap, sz);
	} while (unlikely(!ptr));
	return ptr;
}

static inline void*
_os_alloc2(size_t sz)
{
	void *heap, *ptr;
	if (unlikely(!sz || sz > usable)) {
		__LIBPLATFORM_CLIENT_CRASH__(sz, "Requested allocation size is invalid");
	}
	heap = _os_alloc_heap;
	if (likely(ptr = _os_alloc_alloc(heap, sz))) {
		return ptr;
	}
	return _os_alloc_slow(heap, sz);
}

#pragma mark -
#pragma mark os_alloc_once

typedef struct _os_alloc_once_ctxt_s {
	struct _os_alloc_once_s *slot;
	size_t sz;
	os_function_t init;
} _os_alloc_once_ctxt_s;

static void
_os_alloc(void *ctxt)
{
	_os_alloc_once_ctxt_s *c = ctxt;
	c->slot->ptr = _os_alloc2((c->sz + 0xf) & ~0xfu);
	if (c->init) {
		c->init(c->slot->ptr);
	}
}

void*
_os_alloc_once(struct _os_alloc_once_s *slot, size_t sz, os_function_t init)
{
	_os_alloc_once_ctxt_s c = {
		.slot = slot,
		.sz = sz,
		.init = init,
	};
	_os_once(&slot->once, &c, _os_alloc);
	return slot->ptr;
}
