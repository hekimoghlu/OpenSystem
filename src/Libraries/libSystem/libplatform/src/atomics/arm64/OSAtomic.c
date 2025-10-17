/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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
#include <TargetConditionals.h>

#if TARGET_OS_OSX || TARGET_OS_DRIVERKIT

/*
 * This file implements the following functions for the arm64 architecture.
 *
 *		void  OSAtomicFifoEnqueue( OSFifoQueueHead *__list,	void *__new,
 *			size_t __offset);
 *		void* OSAtomicFifoDequeue( OSFifoQueueHead *__list, size_t __offset);
 *
 */

#include <stdio.h>
#include <machine/cpu_capabilities.h>

#include "libkern/OSAtomic.h"
#include "../OSAtomicFifo.h"

typedef void (OSAtomicFifoEnqueue_t)(OSFifoQueueHead *, void *, size_t);
typedef void *(OSAtomicFifoDequeue_t)(OSFifoQueueHead *, size_t);

void OSAtomicFifoEnqueue(OSFifoQueueHead *__list, void *__new, size_t __offset)
{
	void *addr = commpage_pfz_base;
	addr += _COMM_PAGE_TEXT_ATOMIC_ENQUEUE;

	OSAtomicFifoEnqueue_t *OSAtomicFifoEnqueueInternal = SIGN_PFZ_FUNCTION_PTR(addr);

	return OSAtomicFifoEnqueueInternal(__list, __new, __offset);
}

void * OSAtomicFifoDequeue( OSFifoQueueHead *__list, size_t __offset)
{
	void *addr = commpage_pfz_base;
	addr += _COMM_PAGE_TEXT_ATOMIC_DEQUEUE;

	OSAtomicFifoDequeue_t *OSAtomicFifoDequeueInternal = SIGN_PFZ_FUNCTION_PTR(addr);

	return OSAtomicFifoDequeueInternal(__list, __offset);
}

#endif
