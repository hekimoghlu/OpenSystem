/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 12, 2022.
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

//
//  IOHIDDeviceElementContainer.h
//  IOHIDFamily
//
//  Created by dekom on 10/23/18.
//

#ifndef IOHIDDeviceElementContainer_h
#define IOHIDDeviceElementContainer_h

#include "IOHIDElementContainer.h"

class IOHIDDevice;

class IOHIDDeviceElementContainer : public IOHIDElementContainer
{
    OSDeclareDefaultStructors(IOHIDDeviceElementContainer)
    
private:
    struct ExpansionData {
        IOHIDDevice     *owner;
    };
    
    ExpansionData       *_reserved;
    
protected:
    virtual bool init(void *descriptor, IOByteCount length, IOHIDDevice *owner);
    virtual void free() APPLE_KEXT_OVERRIDE;
    
public:
    static IOHIDDeviceElementContainer *withDescriptor(void *descriptor,
                                                       IOByteCount length,
                                                       IOHIDDevice *owner);
    
    IOReturn updateElementValues(IOHIDElementCookie *cookies,
                                 UInt32 cookieCount = 1) APPLE_KEXT_OVERRIDE;
    
    IOReturn postElementValues(IOHIDElementCookie *cookies,
                               UInt32 cookieCount = 1) APPLE_KEXT_OVERRIDE;
};

#endif /* IOHIDDeviceElementContainer_h */
