/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 4, 2024.
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
#ifndef _IOKIT_HID_IOHIDPOINTINGDEVICE_H
#define _IOKIT_HID_IOHIDPOINTINGDEVICE_H

#include "IOHIDDeviceShim.h"
#include "IOHIPointing.h"

class IOHIDPointingDevice : public IOHIDDeviceShim
{
    OSDeclareDefaultStructors( IOHIDPointingDevice )

private:
    bool			_isScrollPresent;
    UInt8			_numButtons;
    UInt32			_resolution;
    IOBufferMemoryDescriptor *	_report;
    IOHIPointing *		_pointing;

protected:

    virtual void free(void) APPLE_KEXT_OVERRIDE;
    virtual bool handleStart( IOService * provider ) APPLE_KEXT_OVERRIDE;
    
public:
    static IOHIDPointingDevice	* newPointingDeviceAndStart(IOService * owner, UInt8 numButtons = 8, UInt32 resolution = 100, bool scroll = false, UInt32 location = 0);
    
    virtual bool initWithLocation( UInt32 location = 0 ) APPLE_KEXT_OVERRIDE;

    virtual IOReturn newReportDescriptor(
                        IOMemoryDescriptor ** descriptor ) const APPLE_KEXT_OVERRIDE;
    
    virtual OSString * newProductString() const APPLE_KEXT_OVERRIDE;
                                                                
    virtual IOReturn getReport( IOMemoryDescriptor * report,
                                 IOHIDReportType      reportType,
                                 IOOptionBits         options ) APPLE_KEXT_OVERRIDE;

    virtual void postMouseEvent(UInt8 buttons, UInt16 x, UInt16 y, UInt8 wheel=0);
    
    inline bool isScrollPresent() {return _isScrollPresent;}
};

#endif /* !_IOKIT_HID_IOHIDPOINTINGDEVICE_H */
