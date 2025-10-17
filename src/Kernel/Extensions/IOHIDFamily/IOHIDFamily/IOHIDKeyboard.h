/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 26, 2024.
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
#ifndef _IOKIT_HID_IOHIDKEYBOARD_H
#define _IOKIT_HID_IOHIDKEYBOARD_H

#include <IOKit/hidsystem/IOHIDTypes.h>
#include "IOHIKeyboard.h"
#include "IOHIDDevice.h"
#include "IOHIDConsumer.h"
#include "IOHIDElement.h"
#include "IOHIDEventService.h"
#include "IOHIDFamilyPrivate.h"

enum {
    kUSB_CAPSLOCKLED_SET = 2,
    kUSB_NUMLOCKLED_SET = 1
};

#define ADB_CONVERTER_LEN       0xff + 1   //length of array def_usb_2_adb_keymap[]
#define APPLE_ADB_CONVERTER_LEN 0xff + 1   //length of array def_usb_apple_2_adb_keymap[]

#if defined(KERNEL) && !defined(KERNEL_PRIVATE)
class __deprecated_msg("Use DriverKit") IOHIDKeyboard : public IOHIKeyboard
#else
class IOHIDKeyboard : public IOHIKeyboard
#endif
{
    OSDeclareDefaultStructors(IOHIDKeyboard)

    IOHIDEventService *     _provider;
    	
	bool					_repeat;
	bool					_resyncLED;
        
    // LED Specific Members
    UInt8                   _ledState;
    thread_call_t           _asyncLEDThread;

    // Scan Code Array Specific Members
    unsigned int            _usb_2_adb_keymap[ADB_CONVERTER_LEN + 1];
    unsigned int            _usb_apple_2_adb_keymap[APPLE_ADB_CONVERTER_LEN + 1];
    
    // FN Key Member
    bool                    _containsFKey;
    bool                    _isDispatcher;
    
    // *** PRIVATE HELPER METHODS ***
    void                    Set_LED_States(UInt8);
    UInt32                  handlerID();

    // *** END PRIVATE HELPER METHODS ***
    
    // static methods for callbacks, the command gate, new threads, etc.
    static void             _asyncLED (OSObject *target);
                                
public:    
    // Allocator
    static IOHIDKeyboard * 	Keyboard(UInt32 supportedModifiers, bool isDispatcher = false);
    
    // IOService methods
    virtual bool            init(OSDictionary * properties = 0) APPLE_KEXT_OVERRIDE;
    virtual bool            start(IOService * provider) APPLE_KEXT_OVERRIDE;
    virtual void            stop(IOService *  provider) APPLE_KEXT_OVERRIDE;
    virtual void            free(void) APPLE_KEXT_OVERRIDE;

    inline bool             isDispatcher() { return _isDispatcher;};

    // IOHIDevice methods
    UInt32                  interfaceID(void) APPLE_KEXT_OVERRIDE;
    UInt32                  deviceType(void) APPLE_KEXT_OVERRIDE;

    // IOHIKeyboard methods
    const unsigned char * 	defaultKeymapOfLength(UInt32 * length) APPLE_KEXT_OVERRIDE;
    void                    setAlphaLockFeedback(bool LED_state) APPLE_KEXT_OVERRIDE;
    void                    setNumLockFeedback(bool LED_state) APPLE_KEXT_OVERRIDE;
    unsigned                getLEDStatus(void) APPLE_KEXT_OVERRIDE;
    IOReturn                setParamProperties( OSDictionary * dict ) APPLE_KEXT_OVERRIDE;

	void                    dispatchKeyboardEvent(
                                AbsoluteTime                timeStamp,
                                UInt32                      usagePage,
                                UInt32                      usage,
                                bool                        keyDown,
                                IOOptionBits                options = 0);

};


#endif /* !_IOKIT_HID_IOHIDKEYBOARD_H */
