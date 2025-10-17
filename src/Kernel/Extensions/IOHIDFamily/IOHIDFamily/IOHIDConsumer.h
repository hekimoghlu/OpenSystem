/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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
#ifndef _IOKIT_HID_IOHIDCONSUMER_H
#define _IOKIT_HID_IOHIDCONSUMER_H

#include <IOKit/IOLib.h>
#include <IOKit/IOService.h>

// HID system includes.
#include <IOKit/hidsystem/IOHIDDescriptorParser.h>
#include <IOKit/hidsystem/IOHIDShared.h>
#include "IOHIKeyboard.h"
#include "IOHIDKeyboard.h"
#include "IOHIDEventService.h"

// extra includes.
#include <libkern/OSByteOrder.h>

//====================================================================================================
//	IOHIDConsumer
//	Generic driver for usb devices that contain special keys.
//====================================================================================================

#if defined(KERNEL) && !defined(KERNEL_PRIVATE)
class __deprecated_msg("Use DriverKit") IOHIDConsumer : public IOHIKeyboard
#else
class IOHIDConsumer : public IOHIKeyboard
#endif
{
    OSDeclareDefaultStructors(IOHIDConsumer)
    
    IOHIDKeyboard *         _keyboardNub;
    
    UInt32                  _otherEventFlags;
    UInt32                  _cachedEventFlags;
    bool                    _otherCapsLockOn;
	
	bool					_repeat;
    
    bool                    _isDispatcher;
    
    // Our implementation specific stuff.
    UInt32                  findKeyboardsAndGetModifiers();
    
public:
    // Allocator
    static IOHIDConsumer * 		Consumer(bool isDispatcher = false);
    
    // IOService methods
    virtual bool			init(OSDictionary *properties=0) APPLE_KEXT_OVERRIDE;
    virtual bool			start(IOService * provider) APPLE_KEXT_OVERRIDE;
    virtual void			stop(IOService * provider) APPLE_KEXT_OVERRIDE;
    
    virtual void            dispatchConsumerEvent(
                                IOHIDKeyboard *             sendingkeyboardNub,
                                AbsoluteTime                timeStamp,
                                UInt32                      usagePage,
                                UInt32                      usage,
                                UInt32						value,
                                IOOptionBits                options = 0);

    inline bool             isDispatcher() { return _isDispatcher;};
   
    // IOHIKeyboard methods
    virtual const unsigned char*	defaultKeymapOfLength( UInt32 * length ) APPLE_KEXT_OVERRIDE;
    virtual bool                    doesKeyLock(unsigned key) APPLE_KEXT_OVERRIDE;
    virtual unsigned                eventFlags(void) APPLE_KEXT_OVERRIDE;
    virtual unsigned                deviceFlags(void) APPLE_KEXT_OVERRIDE;
    virtual void                    setDeviceFlags(unsigned flags) APPLE_KEXT_OVERRIDE;
    virtual void                    setNumLock(bool val) APPLE_KEXT_OVERRIDE;
    virtual bool                    numLock(void) APPLE_KEXT_OVERRIDE;
    virtual bool                    alphaLock(void) APPLE_KEXT_OVERRIDE;
};
#endif /* !_IOKIT_HID_IOHIDCONSUMER_H */
