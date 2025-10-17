/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 23, 2022.
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
#ifndef _IOKITUSER_IODATAQUEUE_PRIVATE_H
#define _IOKITUSER_IODATAQUEUE_PRIVATE_H

#include <sys/cdefs.h>

__BEGIN_DECLS

#include <IOKit/IODataQueueClient.h>

enum {
    kIODataQueueDeliveryNotificationForce         = (1<<0),
    kIODataQueueDeliveryNotificationSuppress      = (1<<1)
};

typedef uint32_t (*IODataQueueClientEnqueueReadBytesCallback)(void * refcon, void *data, uint32_t dataSize);

IOReturn
_IODataQueueEnqueueWithReadCallback(IODataQueueMemory *dataQueue, uint64_t queueSize, mach_msg_header_t *msgh, uint32_t dataSize, IODataQueueClientEnqueueReadBytesCallback callback, void * refcon);

IOReturn
_IODataQueueEnqueueWithReadCallbackOptions(IODataQueueMemory *dataQueue, uint64_t queueSize, mach_msg_header_t *msgh, uint32_t dataSize, IODataQueueClientEnqueueReadBytesCallback callback, void * refcon, uint32_t options);

/*
 * Internal Peek and Dequeue functions. These functions allow us to pass in the
 * queue size, since queue size is part of shared memory and may be modified by
 * a bad client. See rdar://42664354&42630424&42679853
 */
IODataQueueEntry *_IODataQueuePeek(IODataQueueMemory *dataQueue, uint64_t queueSize,  size_t *entrySize);

IOReturn _IODataQueueDequeue(IODataQueueMemory *dataQueue, uint64_t queueSize, void *data, uint32_t *dataSize);

IOReturn _IODataQueueSendDataAvailableNotification(IODataQueueMemory *dataQueue, mach_msg_header_t *msgh);


__END_DECLS

#endif /* _IOKITUSER_IODATAQUEUE_PRIVATE_H */
