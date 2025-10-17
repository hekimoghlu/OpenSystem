/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 27, 2024.
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
#ifndef _IOHITABLETPOINTER_H
#define _IOHITABLETPOINTER_H

#include <IOKit/hidsystem/IOHIDevice.h>
#include <IOKit/hidsystem/IOLLEvent.h>
#include "IOHITablet.h"

#define kIOHITabletPointerID			"PointerID"
#define kIOHITabletPointerDeviceID		"DeviceID"
#define kIOHITabletPointerVendorType	"VendorPointerType"
#define kIOHITabletPointerType			"PointerType"
#define kIOHITabletPointerSerialNumber	"SerialNumber"
#define kIOHITabletPointerUniqueID		"UniqueID"

#if defined(KERNEL) && !defined(KERNEL_PRIVATE)
class __deprecated_msg("Use DriverKit") IOHITabletPointer : public IOHIDevice
#else
class IOHITabletPointer : public IOHIDevice
#endif
{
    OSDeclareDefaultStructors(IOHITabletPointer);

public:
    IOHITablet	*_tablet;
    UInt16		_deviceID;
    
    static UInt16 generateDeviceID();

    virtual bool init(OSDictionary *propTable) APPLE_KEXT_OVERRIDE;
    virtual bool attach(IOService *provider) APPLE_KEXT_OVERRIDE;

    virtual void dispatchTabletEvent(NXEventData *tabletEvent,
                                     AbsoluteTime ts);
    virtual void dispatchProximityEvent(NXEventData *proximityEvent,
                                        AbsoluteTime ts);
};

#endif /* !_IOHITABLETPOINTER_H */
