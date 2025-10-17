/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 19, 2025.
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
#ifndef _IOKIT_HID_IOHIDFAMILYPRIVATE_H
#define _IOKIT_HID_IOHIDFAMILYPRIVATE_H

#ifdef KERNEL
#include "IOHIDKeys.h"
#include "IOHIDDevice.h"
#endif

#include <pexpert/pexpert.h>

__BEGIN_DECLS

#ifdef KERNEL
bool CompareProperty(IOService * owner, OSDictionary * matching, const char * key, SInt32 * score, SInt32 increment = 0);
bool CompareDeviceUsage( IOService * owner, OSDictionary * matching, SInt32 * score, SInt32 increment = 0);
bool CompareDeviceUsagePairs(IOService * owner, OSDictionary * matching, SInt32 * score, SInt32 increment = 0);
bool CompareProductID( IOService * owner, OSDictionary * matching, SInt32 * score);
bool MatchPropertyTable(IOService * owner, OSDictionary * table, SInt32 * score);
bool CompareNumberPropertyMask( IOService *owner, OSDictionary *matching, const char *key, const char *maskKey, SInt32 *score, SInt32 increment);
bool CompareNumberPropertyArray( IOService * owner, OSDictionary * matching, const char * arrayName, const char * key, SInt32 * score, SInt32 increment);
bool CompareNumberPropertyArrayWithMask( IOService * owner, OSDictionary * matching, const char * arrayName, const char * key, const char * maskKey, SInt32 * score, SInt32 increment);

#define     kEjectKeyDelayMS        0       // the delay for a dedicated eject key
#define     kEjectF12DelayMS        250     // the delay for an F12/eject key

void IOHIDSystemActivityTickle(SInt32 nxEventType, IOService *sender);

void handle_stackshot_keychord(uint32_t keycode);

#define NX_HARDWARE_TICKLE  (NX_LASTEVENT+1)

#define kIOHIDDeviceWillTerminate     iokit_family_msg(sub_iokit_hidsystem, 8)


/*!
 * @method IsIOHIDRestrictedIOKitProperty
 *
 * @abstract
 * Checks OSSymbols to see if they are restricted IOKit properties that should not be allowed to be set.
 *
 * @param key
 * The key that is trying to be set
 *
 * @result
 * Returns true if the property is restricted and should not be set by userland clients. Returns false otherwise.
 */
bool IsIOHIDRestrictedIOKitProperty(const OSSymbol* key);

/*!
 * @method IsIOHIDRestrictedIOKitPropertyDictionary
 *
 * @abstract
 * Checks OSDictionaries for restricted IOKit properties that should not be allowed to be set.
 *
 * @param properties
 * The dictionary that needs to be checked
 *
 * @result
 * Returns true if the any property is restricted in the dictionary and should not be set by userland clients. Returns false otherwise.
 */
bool IsIOHIDRestrictedIOKitPropertyDictionary(const OSDictionary* properties);
#endif

bool isSingleUser();

/*!
* @method getFixedValue
*
* @abstract
* convert length in cm to IOFixed millimeter for given unit exponent.
*
* @discussion
* As per HID spec  unit and unit exponent can be associated with given value.
* Example :
* Value : 0x100, Unit : 0x11, Unit Exponent : 0xE represents 25.6 mm (256 * 10â€“2 cm).
* This function converts value based on unit and unit exponent to IOFixed. Currently
* it only supports unit as length in cm.
*
* @param value
* number that need to be converted to IOFixed
*
* @param unit
* associated unit as per HID spec.
*
* @param exponent
* associated unit exponent as per HID spec.
*
* @result
* Returns IOFixed conversion of number in cm scale if exponent is valid (<=0xF), otherwise will return given value in IOFixed.
*/
IOFixed getFixedValue(uint32_t value, uint32_t unit, uint32_t exponent);

#define kHIDDtraceDebug "hid_dtrace_debug"

typedef enum {
    kHIDTraceGetReport = 1,
    kHIDTraceSetReport,
    kHIDTraceHandleReport,
} HIDTraceFunctionType;

__attribute__((optnone)) __attribute__((unused)) static uint32_t gIOHIDFamilyDtraceDebug()
{
    static uint32_t debug = 0xffffffff;
    
    if (debug == 0xffffffff) {
        debug = 0;
        
        if (!PE_parse_boot_argn(kHIDDtraceDebug, &debug, sizeof (debug))) {
            debug = 0;
        }
    }
    
    return debug;
}

/*!
 used as dtrace probe
 funcType : 1(getReport), 2(setReport), 3(handleReport),
 */
void hid_trace(HIDTraceFunctionType functionType, uintptr_t arg1, uintptr_t arg2, uintptr_t arg3, uintptr_t arg4, uintptr_t arg5);

/*!
 * @define kIOHIDMessageInterfaceRematch
 *
 * @abstract
 * Message from an IOHIDDevice to tell an IOHIDInterface to terminate any clients and rematch
 *
 */
#define kIOHIDMessageInterfaceRematch iokit_vendor_specific_msg(3)

__END_DECLS

#endif /* !_IOKIT_HID_IOHIDFAMILYPRIVATE_H */
