/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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
#ifndef _IOKIT_HID_APPLEEMBEDDEDKEYBOARD_H
#define _IOKIT_HID_APPLEEMBEDDEDKEYBOARD_H

#include <IOKit/hidevent/IOHIDEventDriver.h>
#include "IOHIDPrivateKeys.h"

// Moved up to allow subclasses to use the same keys

enum {
    kSecondaryKeyFnFunction         = 0x01,
    kSecondaryKeyFnKeyboard         = 0x02,
    kSecondaryKeyNumLockKeyboard    = 0x04
};

typedef struct _SecondaryKey {
    UInt8	bits;
    UInt8	swapping;
    UInt16	fnFunctionUsagePage;
    UInt16	fnFunctionUsage;
    UInt16	fnKeyboardUsagePage;
    UInt16	fnKeyboardUsage;
    UInt16	numLockKeyboardUsagePage;
    UInt16	numLockKeyboardUsage;
} SecondaryKey;

class AppleEmbeddedKeyboard: public IOHIDEventDriver
{
    OSDeclareDefaultStructors( AppleEmbeddedKeyboard )
    
//    bool                    _fnKeyDownPhysical;
//    bool                    _fnKeyDownVirtual;
//    bool                    _numLockDown;
//    bool                    _virtualMouseKeysSupport;
//    UInt32                  _fKeyMode;
//    SecondaryKey            _secondaryKeys[255];
//    IOHIDElement *          _keyboardRollOverElement;
    OSDictionary *          _keyboardMap;

//    void                    findKeyboardRollOverElement(OSArray * reportElements);
//        
//    void                    parseSecondaryUsages();
//    
//    bool                    filterSecondaryFnFunctionUsage(
//                                UInt32 *                    usagePage,
//                                UInt32 *                    usage,
//                                bool                        down);
//                                
//    bool                    filterSecondaryFnKeyboardUsage(
//                                UInt32 *                    usagePage,
//                                UInt32 *                    usage,
//                                bool                        down);
//                                
//    bool                    filterSecondaryNumLockKeyboardUsage(
//                                UInt32 *                    usagePage,
//                                UInt32 *                    usage,
//                                bool                        down);
//    
    bool                    filterKeyboardUsage(
                                UInt32 *                    usagePage,
                                UInt32 *                    usage,
                                bool                        down);

protected:
        
    virtual bool            handleStart( IOService * provider ) APPLE_KEXT_OVERRIDE;
    
    virtual IOReturn        setElementValue (
                                UInt32                      usagePage,
                                UInt32                      usage,
                                UInt32                      value ) APPLE_KEXT_OVERRIDE;

    virtual void            dispatchKeyboardEvent(
                                AbsoluteTime                timeStamp,
                                UInt32                      usagePage,
                                UInt32                      usage,
                                UInt32                      value,
                                IOOptionBits                options = 0 ) APPLE_KEXT_OVERRIDE;

public:
    virtual bool            init(OSDictionary * properties = 0) APPLE_KEXT_OVERRIDE;
    virtual void            free(void) APPLE_KEXT_OVERRIDE;

    virtual IOReturn        setSystemProperties( OSDictionary * properties ) APPLE_KEXT_OVERRIDE;

};

#endif /* !_IOKIT_HID_APPLEEMBEDDEDKEYBOARD_H */
