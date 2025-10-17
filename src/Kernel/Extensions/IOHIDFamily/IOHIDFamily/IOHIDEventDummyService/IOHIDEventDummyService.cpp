/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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
#include "IOHIDEventDummyService.h"

#include <AssertMacros.h>
#include "IOHIDDebug.h"

OSDefineMetaClassAndStructors(IOHIDEventDummyService, IOHIDEventService);

bool IOHIDEventDummyService::handleStart(IOService* provider)
{
    bool result = false;
    _interface = OSDynamicCast(IOHIDInterface, provider);
    require(_interface, exit);

    require_action(_interface->open(this, 0,
                                   nullptr,
                                   nullptr),
                   exit,
                   HIDLogError("%s:0x%llx: failed to open %s:0x%llx",
                               getName(), getRegistryEntryID(), _interface->getName(), _interface->getRegistryEntryID()));
    result = true;
exit:
    return result;
}

bool IOHIDEventDummyService::didTerminate(IOService *provider, IOOptionBits options, bool *defer) {
    if (_interface) {
        _interface->close(this);
    }
    _interface = NULL;

    return IOHIDEventService::didTerminate(provider, options, defer);
}


