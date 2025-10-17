/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 2, 2022.
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
#ifndef _OSATOMICQUEUE_H_
#define _OSATOMICQUEUE_H_

#include    <stddef.h>
#include    <sys/cdefs.h>
#include    <stdint.h>
#include    <stdbool.h>
#include    "OSAtomicDeprecated.h"

#include    <Availability.h>

/*! @header Lockless atomic enqueue and dequeue
 * These routines manipulate singly-linked LIFO lists.
 */

__BEGIN_DECLS

/*! @abstract The data structure for a queue head.
    @discussion
	You should always initialize a queue head structure with the
	initialization vector {@link OS_ATOMIC_QUEUE_INIT} before use.
 */
#if defined(__LP64__)

typedef volatile struct {
	void	*opaque1;
	long	 opaque2;
} __attribute__ ((aligned (16))) OSQueueHead;

#else

typedef volatile struct {
	void	*opaque1;
	long	 opaque2;
} OSQueueHead;

#endif

/*! @abstract The initialization vector for a queue head. */
#define	OS_ATOMIC_QUEUE_INIT	{ NULL, 0 }

/*! @abstract Enqueue an element onto a list.
    @discussion
	Memory barriers are incorporated as needed to permit thread-safe access
	to the queue element.
    @param __list
	The list on which you want to enqueue the element.
    @param __new
	The element to add.
    @param __offset
	The "offset" parameter is the offset (in bytes) of the link field
	from the beginning of the data structure being queued (<code>__new</code>).
	The link field should be a pointer type.
	The <code>__offset</code> value needs to be same for all enqueuing and
	dequeuing operations on the same list, even if different structure types
	are enqueued on that list.  The use of <code>offsetset()</code>, defined in
	<code>stddef.h</code> is the common way to specify the <code>__offset</code>
	value.
 */
__OSX_AVAILABLE_STARTING(__MAC_10_5, __IPHONE_4_0)
void  OSAtomicEnqueue( OSQueueHead *__list, void *__new, size_t __offset);


/*! @abstract Dequeue an element from a list.
    @discussion
	Memory barriers are incorporated as needed to permit thread-safe access
	to the queue element.
    @param __list
	The list from which you want to dequeue an element.
    @param __offset
	The "offset" parameter is the offset (in bytes) of the link field
	from the beginning of the data structure being dequeued (<code>__new</code>).
	The link field should be a pointer type.
	The <code>__offset</code> value needs to be same for all enqueuing and
	dequeuing operations on the same list, even if different structure types
	are enqueued on that list.  The use of <code>offsetset()</code>, defined in
	<code>stddef.h</code> is the common way to specify the <code>__offset</code>
	value.
	IMPORTANT: the memory backing the link field of a queue element must not be
	unmapped after OSAtomicDequeue() returns until all concurrent calls to
	OSAtomicDequeue() for the same list on other threads have also returned,
	as they may still be accessing that memory location.
    @result
	Returns the most recently enqueued element, or <code>NULL</code> if the
	list is empty.
 */
__OSX_AVAILABLE_STARTING(__MAC_10_5, __IPHONE_4_0)
void* OSAtomicDequeue( OSQueueHead *__list, size_t __offset);

__END_DECLS

#endif /* _OSATOMICQUEUE_H_ */
