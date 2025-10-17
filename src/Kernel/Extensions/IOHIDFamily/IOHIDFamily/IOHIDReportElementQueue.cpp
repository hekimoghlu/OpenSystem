/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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
#include <IOKit/system.h>
#include <IOKit/IOLib.h>
#include <AssertMacros.h>
#include <IOKit/IODataQueueShared.h>
#include "IOHIDDebug.h"
#include "IOHIDReportElementQueue.h"

#define super IOHIDEventQueue

OSDefineMetaClassAndStructors(IOHIDReportElementQueue, super)

IOHIDReportElementQueue *IOHIDReportElementQueue::withCapacity(UInt32 size, IOHIDLibUserClient *client)
{
    IOHIDReportElementQueue *queue = new IOHIDReportElementQueue;
    UInt32 paddedSize = 0;
    if (queue) {
        queue->fClient = client;
    }

    if (size < HID_QUEUE_CAPACITY_MIN) {
        size = HID_QUEUE_CAPACITY_MIN;
    }

    if (os_add_overflow(size, sizeof(IOHIDQueueHeader), &paddedSize)) {
        HIDLogError("IOHIDReportElementQueue: invalid queue size, too large: %u", (unsigned int)size);
        return NULL;
    }

    if (queue && !queue->initWithCapacity(paddedSize)) {
        queue->release();
        queue = NULL;
    }

    if (queue) {
        queue->header = (IOHIDQueueHeader*)queue->dataQueue;
        queue->dataQueue = reinterpret_cast<IODataQueueMemory*>(reinterpret_cast<uint8_t*>(queue->dataQueue) + sizeof(IOHIDQueueHeader));
        queue->setQueueSize(size);
        queue->dataQueue->queueSize = size;
    }

    return queue;
}

Boolean IOHIDReportElementQueue::enqueue(IOHIDElementValue *value)
{
    return (fClient->processElement(value, this) == kIOReturnSuccess);
}

Boolean IOHIDReportElementQueue::enqueue(void *data, UInt32 dataSize)
{
    Boolean result;

    result = super::enqueue(data, dataSize);

    return result;
}

void IOHIDReportElementQueue::setPendingReports()
{
    header->status |= kIOHIDQueueStatusBlocked;
    sendDataAvailableNotification();
}

bool IOHIDReportElementQueue::pendingReports()
{
    return header->status & kIOHIDQueueStatusBlocked;
}

void IOHIDReportElementQueue::clearPendingReports()
{
    header->status &= ~kIOHIDQueueStatusBlocked;
}

void IOHIDReportElementQueue::free()
{
    // Fixup dataQueue pointer and size, so that superclass can free the memory for us.
    if (header) {
        setQueueSize(getQueueSize() + sizeof(IOHIDQueueHeader));
        dataQueue = (IODataQueueMemory*)header;
    }
    super::free();
}

IOMemoryDescriptor * 
IOHIDReportElementQueue::getMemoryDescriptor()
{
    IOMemoryDescriptor *descriptor = NULL;

    if (header != NULL) {
        descriptor = IOMemoryDescriptor::withAddress(header, getQueueSize() + DATA_QUEUE_MEMORY_HEADER_SIZE + DATA_QUEUE_MEMORY_APPENDIX_SIZE + sizeof(IOHIDQueueHeader), kIODirectionOutIn);
    }

    return descriptor;
}

bool
IOHIDReportElementQueue::serialize(OSSerialize * serializer) const
{
    bool ret = false;
    
    if (serializer->previouslySerialized(this)) {
        return true;
    }
    
    OSDictionary *dict = OSDictionary::withCapacity(2);
    if (dict) {
        OSNumber *num = OSNumber::withNumber(dataQueue->head, 32);
        if (num) {
            dict->setObject("head", num);
            num->release();
        }
        num = OSNumber::withNumber(dataQueue->tail, 32);
        if (num) {
            dict->setObject("tail", num);
            num->release();
        }
        num = OSNumber::withNumber(_enqueueErrorCount, 64);
        if (num) {
            dict->setObject("EnqueueErrorCount", num);
            num->release();
        }
        num = OSNumber::withNumber(_reserved->queueSize, 64);
        if (num) {
            dict->setObject("QueueSize", num);
            num->release();
        }
        num = OSNumber::withNumber(_numEntries, 64);
        if (num) {
            dict->setObject("numEntries", num);
            num->release();
        }
        num = OSNumber::withNumber(_entrySize, 64);
        if (num) {
            dict->setObject("entrySize", num);
            num->release();
        }
        OSDictionary * usageDict = copyUsageCountDict();
        if (usageDict) {
            dict->setObject("UsagePercentHist", usageDict);
            OSSafeReleaseNULL(usageDict);
        }
        num = OSNumber::withNumber(header->status, 32);
        if (num) {
            dict->setObject("status", num);
            num->release();
        }
        ret = dict->serialize(serializer);
        dict->release();
    }
    
    return ret;
}
