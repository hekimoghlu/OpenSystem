/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 5, 2025.
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
#ifndef _IOKIT_HID_IOHIDEVENTQUEUE_H
#define _IOKIT_HID_IOHIDEVENTQUEUE_H

#include <IOKit/IOSharedDataQueue.h>
#include <IOKit/IOLocks.h>
#include "IOHIDKeys.h"
#include "IOHIDElementPrivate.h"
#include "IOHIDLibUserClient.h"

enum {
    kHIDQueueStarted    = 0x01,
    kHIDQueueDisabled   = 0x02
};

#define HID_QUEUE_HEADER_SIZE               (sizeof(IOHIDElementValue))  // 24b
#define HID_QUEUE_CAPACITY_MIN              16384       // 16k
#define HID_QUEUE_CAPACITY_MAX              131072      // 128k

#define HID_QUEUE_USAGE_BUCKETS 11
#define HID_QUEUE_BUCKET_DENOM (100/(HID_QUEUE_USAGE_BUCKETS-1))

//---------------------------------------------------------------------------
// IOHIDEventQueue class.
//
// IOHIDEventQueue is a subclass of IOSharedDataQueue. But this may change
// if the HID Manager requires HID specific functionality for the
// event queueing.

class IOHIDEventQueue: public IOSharedDataQueue
{
    OSDeclareDefaultStructors( IOHIDEventQueue )
    
protected:
    UInt32                  _numEntries;
    UInt32                  _entrySize;
    IOOptionBits            _state;
    IOHIDQueueOptionsType   _options;
    UInt64                  _enqueueErrorCount;
    UInt64                  _usageCounts[HID_QUEUE_USAGE_BUCKETS];

    void            updateUsageCounts();
    OSDictionary    *copyUsageCountDict() const;

public:
    static IOHIDEventQueue *withCapacity(UInt32 size);
    static IOHIDEventQueue *withEntries(UInt32 numEntries, UInt32 entrySize);
    
    virtual Boolean enqueue(void *data, UInt32 dataSize) APPLE_KEXT_OVERRIDE;
    
    inline virtual void setOptions(IOHIDQueueOptionsType flags) { _options = flags; }
    inline virtual IOHIDQueueOptionsType getOptions() { return _options; }
    
    // start/stop are accessible from user space.
    inline virtual void start() { _state |= kHIDQueueStarted; }
    inline virtual void stop() { _state &= ~kHIDQueueStarted; }
    
    // enable disable are only accessible from kernel.
    inline virtual void enable() { _state &= ~kHIDQueueDisabled; }
    inline virtual void disable() { _state |= kHIDQueueDisabled; }
    
    virtual bool serialize(OSSerialize * serializer) const APPLE_KEXT_OVERRIDE;

};

#endif /* !_IOKIT_HID_IOHIDEVENTQUEUE_H */
