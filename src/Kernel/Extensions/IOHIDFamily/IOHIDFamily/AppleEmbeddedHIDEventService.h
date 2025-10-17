/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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
#ifndef _IOKIT_HID_APPLEEMBEDDEDHIDEVENTSERVICE_H
#define _IOKIT_HID_APPLEEMBEDDEDHIDEVENTSERVICE_H

#include <IOKit/hidevent/IOHIDEventService.h>
#include <IOKit/hid/AppleEmbeddedHIDKeys.h>

class AppleEmbeddedHIDEventService: public IOHIDEventService
{
    OSDeclareAbstractStructors( AppleEmbeddedHIDEventService )

public:
    virtual bool handleStart(IOService * provider) APPLE_KEXT_OVERRIDE;

protected:
    virtual void dispatchAccelerometerEvent(AbsoluteTime timestamp, IOFixed x, IOFixed y, IOFixed z, IOHIDMotionType type = 0, IOHIDMotionPath subType = 0, UInt32 sequence = 0, IOOptionBits options=0);

    virtual void dispatchGyroEvent(AbsoluteTime timestamp, IOFixed x, IOFixed y, IOFixed z, IOHIDMotionType type = 0, IOHIDMotionPath subType = 0, UInt32 sequence = 0, IOOptionBits options=0);

    virtual void dispatchCompassEvent(AbsoluteTime timestamp, IOFixed x, IOFixed y, IOFixed z, IOHIDMotionType type=0, IOHIDMotionPath subType = 0, UInt32 sequence = 0, IOOptionBits options=0);
    
    virtual void dispatchProximityEvent(AbsoluteTime timestamp, IOHIDProximityDetectionMask mask, UInt32 level = 0, IOOptionBits options=0);

    virtual void dispatchAmbientLightSensorEvent(AbsoluteTime timestamp, UInt32 level, UInt32 channel0 = 0, UInt32 channel1 = 0, UInt32 channel2 = 0, UInt32 channel3 = 0, IOOptionBits options=0);
 
    virtual void dispatchTemperatureEvent(AbsoluteTime timestamp, IOFixed temperature, IOOptionBits options=0);

    virtual void dispatchPowerEvent(AbsoluteTime timestamp, int64_t measurement, IOHIDPowerType powerType, IOHIDPowerSubType powerSubType = 0, IOOptionBits options=0);

    virtual void dispatchVendorDefinedEvent(AbsoluteTime timeStamp, UInt32 usagePage, UInt32 usage, UInt32 version, UInt8 * data, UInt32 length, IOOptionBits options = 0);
    
    virtual void dispatchBiometricEvent(AbsoluteTime timeStamp, IOFixed level, IOHIDBiometricEventType eventType, IOOptionBits options = 0) APPLE_KEXT_OVERRIDE;

    virtual void dispatchAtmosphericPressureEvent(AbsoluteTime timeStamp, IOFixed level, UInt32 sequence=0, IOOptionBits options=0);

    virtual IOHIDOrientationType getOrientation();

    virtual IOHIDPlacementType getPlacement();

    virtual UInt32          getReportInterval() APPLE_KEXT_OVERRIDE;
 
    virtual void dispatchAmbientLightSensorEvent(AbsoluteTime timestamp, UInt32 level, IOHIDEventColorSpace colorSpace, IOHIDDouble colorComponent0, IOHIDDouble colorComponent1, IOHIDDouble colorComponent2, IOOptionBits options);

 //   virtual void dispatchBiometricEvent(AbsoluteTime timeStamp, IOFixed level, IOHIDBiometricEventType eventType, UInt32 usagePage, UInt32 usage, UInt8 tapCount = 1, IOOptionBits options = 0);
};

#endif /* _IOKIT_HID_APPLEEMBEDDEDHIDEVENTSERVICE_H */

