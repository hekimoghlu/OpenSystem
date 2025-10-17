/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 30, 2023.
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
#ifndef _IOKIT_HID_IOHIDEVENTOVERRIDEDRIVER_H
#define _IOKIT_HID_IOHIDEVENTOVERRIDEDRIVER_H

#include <IOKit/hidevent/IOHIDEventDriver.h>

class IOHIDEvent;

/*! @class IOHIDEventOverrideDriver : public IOHIDEventDriver
    @abstract
    @discussion
*/

class IOHIDEventOverrideDriver: public IOHIDEventDriver
{
    OSDeclareDefaultStructors( IOHIDEventOverrideDriver )
    
private:
    uint32_t    _rawPointerButtonMask;
    uint32_t    _resultantPointerButtonMask;

    struct {
        IOHIDEventType  eventType;
        union {
            struct {
                uint32_t        usagePage;
                uint32_t        usage;
            } keyboard;
            struct {
                uint32_t        mask;
            } pointer;
        } u;
    } _buttonMap[32];
    
protected:
    virtual void dispatchEvent(IOHIDEvent * event, IOOptionBits options=0) APPLE_KEXT_OVERRIDE;
    
public:
    virtual bool handleStart( IOService * provider ) APPLE_KEXT_OVERRIDE;
};

#endif /* !_IOKIT_HID_IOHIDEVENTOVERRIDEDRIVER_H */
