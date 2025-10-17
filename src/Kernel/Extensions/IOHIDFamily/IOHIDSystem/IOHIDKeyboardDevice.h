/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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
#ifndef _IOKIT_HID_IOHIDKEYBOARDDEVICE_H
#define _IOKIT_HID_IOHIDKEYBOARDDEVICE_H

#include "IOHIDDeviceShim.h"
#include "IOHIKeyboard.h"

class IOHIDKeyboardDevice : public IOHIDDeviceShim
{
    OSDeclareDefaultStructors( IOHIDKeyboardDevice )

private:
    IOBufferMemoryDescriptor *	_report;
    IOHIKeyboard *		_keyboard;
    
    UInt8			_cachedLEDState;
    UInt32          _lastFlags;
    
    bool			_inputReportOnly;

protected:

    virtual void free(void) APPLE_KEXT_OVERRIDE;
    virtual bool handleStart( IOService * provider ) APPLE_KEXT_OVERRIDE;
    
public:
    static IOHIDKeyboardDevice	* newKeyboardDeviceAndStart(IOService * owner, UInt32 location = 0);
    
    virtual bool initWithLocation( UInt32 location = 0 ) APPLE_KEXT_OVERRIDE;

    virtual IOReturn newReportDescriptor(
                        IOMemoryDescriptor ** descriptor ) const APPLE_KEXT_OVERRIDE;
                        
    virtual OSString * newProductString() const APPLE_KEXT_OVERRIDE;

    virtual IOReturn getReport( IOMemoryDescriptor * report,
                                 IOHIDReportType      reportType,
                                 IOOptionBits         options ) APPLE_KEXT_OVERRIDE;
                                 
    virtual IOReturn setReport( IOMemoryDescriptor * report,
                                IOHIDReportType      reportType,
                                IOOptionBits         options ) APPLE_KEXT_OVERRIDE;
                                                                
    virtual void postKeyboardEvent(UInt8 key, bool keyDown);
    virtual void postFlagKeyboardEvent(UInt32 flags);
    
    virtual void setCapsLockLEDElement(bool state);
    virtual void setNumLockLEDElement(bool state);
  
    virtual bool matchPropertyTable(
                                    OSDictionary *              table,
                                    SInt32 *                    score) APPLE_KEXT_OVERRIDE;
  
};

#endif /* !_IOKIT_HID_IOHIDKEYBOARDDEVICE_H */
