/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 8, 2023.
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
#include "IOHITabletPointer.h"

OSDefineMetaClassAndStructors(IOHITabletPointer, IOHIDevice)

UInt16 IOHITabletPointer::generateDeviceID()
{
    static UInt16 _nextDeviceID = 0;
    return _nextDeviceID++;
}

bool IOHITabletPointer::init( OSDictionary *propTable )
{
    if (!IOHIDevice::init(propTable)) {
        return false;
    }

    _deviceID = generateDeviceID();
    setProperty(kIOHITabletPointerDeviceID, (unsigned long long)_deviceID, 16);

    return true;
}

bool IOHITabletPointer::attach( IOService * provider )
{
    if (!IOHIDevice::attach(provider)) {
        return false;
    }

    _tablet = OSDynamicCast(IOHITablet, provider);

    return true;
}

void IOHITabletPointer::dispatchTabletEvent(NXEventData *tabletEvent,
                                            AbsoluteTime ts)
{
    if (_tablet) {
        _tablet->dispatchTabletEvent(tabletEvent, ts);
    }
}

void IOHITabletPointer::dispatchProximityEvent(NXEventData *proximityEvent,
                                               AbsoluteTime ts)
{
    if (_tablet) {
        _tablet->dispatchProximityEvent(proximityEvent, ts);
    }
}
