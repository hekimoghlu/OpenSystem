/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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
#include <IOKit/IOLib.h>
#include "AppleEmbeddedKeyboard.h"
#include "AppleHIDUsageTables.h"
#include "IOHIDUsageTables.h"
#include "IOHIDKeyboard.h"
#include "IOHIDPrivateKeys.h"
#include "IOHIDKeys.h"
#include "IOLLEvent.h"

#define super IOHIDEventDriver

OSDefineMetaClassAndStructors( AppleEmbeddedKeyboard, IOHIDEventDriver )

//====================================================================================================
// AppleEmbeddedKeyboard::init
//====================================================================================================
bool AppleEmbeddedKeyboard::init(OSDictionary * properties)
{
    if ( !super::init(properties) )
        return false;
        
    return true;
}

//====================================================================================================
// AppleEmbeddedKeyboard::free
//====================================================================================================
void AppleEmbeddedKeyboard::free()
{
    if ( _keyboardMap )
        _keyboardMap->release();
    
    super::free();
}

//====================================================================================================
// AppleEmbeddedKeyboard::handleStart
//====================================================================================================
bool AppleEmbeddedKeyboard::handleStart( IOService * provider )
{
    setProperty(kIOHIDAppleVendorSupported, kOSBooleanTrue);
    
    if (!super::handleStart(provider))
        return false;
    
    _keyboardMap = OSDynamicCast(OSDictionary, copyProperty(kKeyboardUsageMapKey));
    
    return true;
}


//====================================================================================================
// AppleEmbeddedKeyboard::setElementValue
//====================================================================================================
IOReturn AppleEmbeddedKeyboard::setElementValue (
                                UInt32                      usagePage,
                                UInt32                      usage,
                                UInt32                      value )
{

    return super::setElementValue(usagePage, usage, value);
}

//====================================================================================================
// AppleEmbeddedKeyboard::dispatchKeyboardEvent
//====================================================================================================
void AppleEmbeddedKeyboard::dispatchKeyboardEvent(
                                AbsoluteTime                timeStamp,
                                UInt32                      usagePage,
                                UInt32                      usage,
                                UInt32                      value,
                                IOOptionBits                options)
{
    filterKeyboardUsage(&usagePage, &usage, value);


    super::dispatchKeyboardEvent(timeStamp, usagePage, usage, value, options);
}



//====================================================================================================
// AppleEmbeddedKeyboard::filterKeyboardUsage
//====================================================================================================
bool AppleEmbeddedKeyboard::filterKeyboardUsage(UInt32 *                    usagePage,
                                                UInt32 *                    usage,
                                                bool                        down __unused)
{
    char key[32];
    
    bzero(key, sizeof(key));
    snprintf(key, sizeof(key), "0x%04x%04x", (uint16_t)*usagePage, (uint16_t)*usage);
    
    if ( _keyboardMap ) {
        OSNumber * map = OSDynamicCast(OSNumber, _keyboardMap->getObject(key));
        
        if ( map ) {            
            *usagePage  = (map->unsigned32BitValue()>>16) & 0xffff;
            *usage      = map->unsigned32BitValue()&0xffff;
            
        }
    }
    
    return false;
}

//====================================================================================================
// AppleEmbeddedKeyboard::setSystemProperties
//====================================================================================================
IOReturn AppleEmbeddedKeyboard::setSystemProperties( OSDictionary * properties )
{
    return super::setSystemProperties(properties);
}

//====================================================================================================
