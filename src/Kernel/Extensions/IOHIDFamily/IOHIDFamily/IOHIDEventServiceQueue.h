/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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
#ifndef _IOKIT_HID_IOHIDEVENTSERVICEQUEUE_H
#define _IOKIT_HID_IOHIDEVENTSERVICEQUEUE_H

#include <IOKit/IOSharedDataQueue.h>

enum {
    kIOHIDEventServiceQueueOptionNotificationForce = 0x1,
    kIOHIDEventServiceQueueOptionNoFullNotification = 0x2,
};

class IOHIDEvent;
//---------------------------------------------------------------------------
// IOHIDEventSeviceQueue class.
//
// IOHIDEventServiceQueue is a subclass of IOSharedDataQueue.

class IOHIDEventServiceQueue: public IOSharedDataQueue
{
    OSDeclareDefaultStructors( IOHIDEventServiceQueue )
    
protected:
    IOMemoryDescriptor *    _descriptor;
    Boolean                 _state;
    OSObject                *_owner;
    UInt32                  _options;
    UInt64                  _notificationCount;

    virtual void sendDataAvailableNotification() APPLE_KEXT_OVERRIDE;

public:
    static IOHIDEventServiceQueue *withCapacity(UInt32 size, UInt32 options = 0);
    static IOHIDEventServiceQueue *withCapacity(OSObject *owner, UInt32 size, UInt32 options = 0);
    
    virtual void free(void) APPLE_KEXT_OVERRIDE;
    
    inline Boolean getState() { return _state; }
    inline void setState(Boolean state) { _state = state; }
    
    inline OSObject *getOwner() { return _owner; }

    virtual Boolean enqueueEvent(IOHIDEvent * event);

    virtual IOMemoryDescriptor *getMemoryDescriptor(void) APPLE_KEXT_OVERRIDE;
    virtual void setNotificationPort(mach_port_t port) APPLE_KEXT_OVERRIDE;
    virtual bool serialize(OSSerialize * serializer) const APPLE_KEXT_OVERRIDE;

};

//---------------------------------------------------------------------------
#endif /* !_IOKIT_HID_IOHIDEVENTSERVICEQUEUE_H */
