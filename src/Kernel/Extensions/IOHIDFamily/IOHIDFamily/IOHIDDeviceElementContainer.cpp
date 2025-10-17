/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 11, 2022.
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
//  IOHIDDeviceElementContainer.cpp
//  IOHIDFamily
//
//  Created by dekom on 10/23/18.
//

#include "IOHIDDeviceElementContainer.h"
#include <AssertMacros.h>
#include "IOHIDDevice.h"

#define super IOHIDElementContainer
OSDefineMetaClassAndStructors(IOHIDDeviceElementContainer, IOHIDElementContainer)

#define _owner  _reserved->owner

bool IOHIDDeviceElementContainer::init(void *descriptor,
                                       IOByteCount length,
                                       IOHIDDevice *owner)
{
    bool result = false;
    
    require(super::init(descriptor, length), exit);
    
    _reserved = IOMallocType(ExpansionData);
    require(_reserved, exit);
    
    bzero(_reserved, sizeof(ExpansionData));
    
    _owner = owner;
    
    result = true;
    
exit:
    return result;
}

IOHIDDeviceElementContainer *IOHIDDeviceElementContainer::withDescriptor(
                                                            void *descriptor,
                                                            IOByteCount length,
                                                            IOHIDDevice *owner)
{
    IOHIDDeviceElementContainer *me = new IOHIDDeviceElementContainer;
    
    if (me && !me->init(descriptor, length, owner)) {
        me->release();
        return NULL;
    }
    
    return me;
}

void IOHIDDeviceElementContainer::free()
{
    if (_reserved) {
        IOFreeType(_reserved, ExpansionData);
    }
    
    super::free();
}

IOReturn IOHIDDeviceElementContainer::updateElementValues(
                                                    IOHIDElementCookie *cookies,
                                                    UInt32 cookieCount)
{
    return _owner->updateElementValues(cookies, cookieCount);
}

IOReturn IOHIDDeviceElementContainer::postElementValues(
                                                    IOHIDElementCookie *cookies,
                                                    UInt32 cookieCount)
{
    return _owner->postElementValues(cookies, cookieCount);
}
