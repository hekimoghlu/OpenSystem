/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 19, 2023.
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
#ifndef _IOKIT_HID_IOHIDREPORTELEMENTQUEUE_H
#define _IOKIT_HID_IOHIDREPORTELEMENTQUEUE_H

#include "IOHIDEventQueue.h"
#include "IOHIDLibUserClient.h"

//---------------------------------------------------------------------------
// IODHIDReportElementQueue class.
//
// IODHIDReportElementQueue is a subclass of IOHIDEventQueue. This allows us to enqueue
// large input reports by passing the element into the UC and letting it handle the memory.
// The report is actually enqueued with the call too enqueue(void*, size_t) which puts the
// report into the shared memory.

class IOHIDReportElementQueue: public IOHIDEventQueue
{
    OSDeclareDefaultStructors( IOHIDReportElementQueue )

protected:
    IOHIDLibUserClient *fClient;
    IOHIDQueueHeader *header;

    virtual void free() APPLE_KEXT_OVERRIDE;

public:
    static IOHIDReportElementQueue *withCapacity(UInt32 size, IOHIDLibUserClient *client);

    virtual Boolean enqueue(IOHIDElementValue* element);
    virtual Boolean enqueue(void *data, UInt32 dataSize) APPLE_KEXT_OVERRIDE;
    virtual IOMemoryDescriptor *getMemoryDescriptor() APPLE_KEXT_OVERRIDE;

    virtual bool serialize(OSSerialize * serializer) const APPLE_KEXT_OVERRIDE;

    bool pendingReports();
    void setPendingReports();
    void clearPendingReports();
};

#endif /* !_IOKIT_HID_IOHIDREPORTELEMENTQUEUE_H */
