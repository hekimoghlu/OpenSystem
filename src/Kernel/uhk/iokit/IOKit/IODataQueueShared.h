/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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
#ifndef _IOKIT_IODATAQUEUESHARED_H
#define _IOKIT_IODATAQUEUESHARED_H

#include <libkern/OSTypes.h>
#include <mach/port.h>
#include <mach/message.h>

/*!
 * @typedef IODataQueueEntry
 * @abstract Represents an entry within the data queue
 * @discussion This is a variable sized struct.  The data field simply represents the start of the data region.  The size of the data region is stored in the size field.  The whole size of the specific entry is the size of a UInt32 plus the size of the data region.
 * @field size The size of the following data region.
 * @field data Represents the beginning of the data region.  The address of the data field is a pointer to the start of the data region.
 */
typedef struct _IODataQueueEntry {
	UInt32  size;
	UInt8   data[4];
} IODataQueueEntry;

/*!
 * @typedef IODataQueueMemory
 * @abstract A struct mapping to the header region of a data queue.
 * @discussion This struct is variable sized.  The struct represents the data queue header information plus a pointer to the actual data queue itself.  The size of the struct is the combined size of the header fields (3 * sizeof(UInt32)) plus the actual size of the queue region.  This size is stored in the queueSize field.
 * @field queueSize The size of the queue region pointed to by the queue field.
 * @field head The location of the queue head.  This field is represented as a byte offset from the beginning of the queue memory region.
 * @field tail The location of the queue tail.  This field is represented as a byte offset from the beginning of the queue memory region.
 * @field queue Represents the beginning of the queue memory region.  The size of the region pointed to by queue is stored in the queueSize field.
 */
typedef struct _IODataQueueMemory {
	UInt32            queueSize;
	volatile UInt32   head;
	volatile UInt32   tail;
	IODataQueueEntry  queue[1];
} IODataQueueMemory;

/*!
 * @typedef IODataQueueAppendix
 * @abstract A struct mapping to the appendix region of a data queue.
 * @discussion This struct is variable sized dependent on the version.  The struct represents the data queue appendix information.
 * @field version The version of the queue appendix.
 * @field msgh Mach message header containing the notification mach port associated with this queue.
 */
typedef struct _IODataQueueAppendix {
	UInt32            version;
	mach_msg_header_t msgh;
} IODataQueueAppendix;

/*!
 * @defined DATA_QUEUE_ENTRY_HEADER_SIZE Represents the size of the data queue entry header independent of the actual size of the data in the entry.  This is the overhead of each entry in the queue.  The total size of an entry is equal to this value plus the size stored in the entry's size field (in IODataQueueEntry).
 */
#define DATA_QUEUE_ENTRY_HEADER_SIZE (sizeof(IODataQueueEntry) - 4)

/*!
 * @defined DATA_QUEUE_MEMORY_HEADER_SIZE Represents the size of the data queue memory header independent of the actual size of the queue data itself.  The total size of the queue memory is equal to this value plus the size of the queue appendix and the size of the queue data region which is stored in the queueSize field of IODataQueueMeory.
 */
#define DATA_QUEUE_MEMORY_HEADER_SIZE (sizeof(IODataQueueMemory) - sizeof(IODataQueueEntry))

/*!
 * @defined DATA_QUEUE_MEMORY_APPENDIX_SIZE Represents the size of the data queue memory appendix independent of the actual size of the queue data itself.  The total size of the queue memory is equal to this value plus the size of queue header and size of the queue data region which is stored in the queueSize field of IODataQueueMeory.
 */
#define DATA_QUEUE_MEMORY_APPENDIX_SIZE (sizeof(IODataQueueAppendix))

#endif /* _IOKIT_IODATAQUEUESHARED_H */
