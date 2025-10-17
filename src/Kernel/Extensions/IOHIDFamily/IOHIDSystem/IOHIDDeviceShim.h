/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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
#ifndef _IOKIT_HID_IOHIDDEVICESHIM_H
#define _IOKIT_HID_IOHIDDEVICESHIM_H

#include "IOHIDDevice.h"
#include "IOHIDevice.h"

#define kIOHIDAppleVendorID 1452
typedef enum IOHIDTransport {
    kIOHIDTransportNone = 0,
    kIOHIDTransportUSB,
    kIOHIDTransportADB,
    kIOHIDTransportPS2
} IOHIDTransport;

class IOHIDDeviceShim : public IOHIDDevice
{
    OSDeclareDefaultStructors( IOHIDDeviceShim )

private:
    IOService *       _device;
    IOHIDevice *      _hiDevice;
    IOHIDTransport		_transport;
    UInt32            _location;
    boolean_t         _allowVirtualProvider;

protected:

    virtual bool handleStart( IOService * provider ) APPLE_KEXT_OVERRIDE;
    
public:
    virtual IOReturn newReportDescriptor(
                        IOMemoryDescriptor ** descriptor ) const APPLE_KEXT_OVERRIDE = 0;
    virtual bool initWithLocation(UInt32 location = 0);
    
    virtual IOHIDTransport transport() {return _transport;};
    
    virtual OSString * newTransportString() const APPLE_KEXT_OVERRIDE;
    virtual OSString * newProductString() const APPLE_KEXT_OVERRIDE;
    virtual OSString * newManufacturerString() const APPLE_KEXT_OVERRIDE;
    virtual OSNumber * newVendorIDNumber() const APPLE_KEXT_OVERRIDE;
    virtual OSNumber * newProductIDNumber() const APPLE_KEXT_OVERRIDE;
    virtual OSNumber * newLocationIDNumber() const APPLE_KEXT_OVERRIDE;
    virtual OSString * newSerialNumberString() const APPLE_KEXT_OVERRIDE;
    
    virtual bool       isSeized();
    virtual bool       initWithParameters(UInt32 location, boolean_t allowVirtualProvider);

};

#endif /* !_IOKIT_HID_IOHIDDEVICESHIM_H */
