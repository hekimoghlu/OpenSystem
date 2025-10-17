/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 22, 2025.
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

#include <libkern/OSAtomic.h>
#include <System/i386/cpu_capabilities.h>

#define OS_UNFAIR_LOCK_INLINE 1
#include "os/lock_private.h"

typedef	volatile struct {
	void	*first;
	void	*last;
	os_unfair_lock	 lock;
} __attribute__ ((aligned (16))) UnfairFifoQueueHead;

#define set_next(element, offset, new) \
	*((void**)(((uintptr_t)element) + offset)) = new;
#define get_next(element, offset) \
	*((void**)(((uintptr_t)element) + offset));

// This is a naive implementation using unfair locks to support translated
// x86_64 apps only. Native x86_64 and arm64 apps will use the
// PFZ implementations
void OSAtomicFifoEnqueue$VARIANT$UnfairLock(UnfairFifoQueueHead *list, void *new, size_t offset) {
	set_next(new, offset, NULL);

	os_unfair_lock_lock_inline((os_unfair_lock_t)&list->lock);
	if (list->last == NULL) {
		list->first = new;
	} else {
		set_next(list->last, offset, new);
	}
	list->last = new;
	os_unfair_lock_unlock_inline((os_unfair_lock_t)&list->lock);
}

void* OSAtomicFifoDequeue$VARIANT$UnfairLock(UnfairFifoQueueHead *list, size_t offset) {
	os_unfair_lock_lock_inline((os_unfair_lock_t)&list->lock);
	void *element = list->first;
	if (element != NULL) {
		void *next = get_next(element, offset);
		if (next == NULL) {
			list->last = NULL;
		}
		list->first = next;
	}
	os_unfair_lock_unlock_inline((os_unfair_lock_t)&list->lock);

	return element;
}

#define MakeResolver(name)													\
	void * name ## Resolver(void) __asm__("_" #name);						\
	void * name ## Resolver(void) {											\
		__asm__(".symbol_resolver _" #name);								\
		uint64_t capabilities = *(uint64_t*)_COMM_PAGE_CPU_CAPABILITIES64;	\
		if (capabilities & kIsTranslated) {									\
			return name ## $VARIANT$UnfairLock; 							\
		} else {															\
			return name ## $VARIANT$PFZ;    								\
		}                                                          			\
	}

void OSAtomicFifoEnqueue$VARIANT$PFZ(OSFifoQueueHead *, void *, size_t);
void* OSAtomicFifoDequeue$VARIANT$PFZ(OSFifoQueueHead *, size_t);

MakeResolver(OSAtomicFifoEnqueue)
MakeResolver(OSAtomicFifoDequeue)
