/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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
#include "IOHIDEvent.h"
#include "AppleEmbeddedHIDEventService.h"


//===========================================================================
// AppleEmbeddedHIDEventService class
#define super IOHIDEventService

OSDefineMetaClassAndAbstractStructors( AppleEmbeddedHIDEventService, IOHIDEventService )

//====================================================================================================
// AppleEmbeddedHIDEventService::handleStart
//====================================================================================================
bool AppleEmbeddedHIDEventService::handleStart(IOService * provider)
{
    uint32_t value;
    
    if ( !super::handleStart(provider) )
        return FALSE;
    
    value = getOrientation();
    if ( value )
        setProperty(kIOHIDOrientationKey, value, 32);

    value = getPlacement();
    if ( value )
        setProperty(kIOHIDPlacementKey, value, 32);
    
    // RY: all embedded services are built-in
    setProperty(kIOHIDBuiltInKey, true);
    
    return TRUE;
}


//====================================================================================================
// AppleEmbeddedHIDEventService::dispatchAccelerometerEvent
//====================================================================================================
void AppleEmbeddedHIDEventService::dispatchAccelerometerEvent(AbsoluteTime timestamp, IOFixed x, IOFixed y, IOFixed z, IOHIDMotionType type, IOHIDMotionPath subType, UInt32 sequence, IOOptionBits options)
{
    IOHIDEvent * event = IOHIDEvent::accelerometerEvent(timestamp, x, y, z, type, subType, sequence, options);
    
    if ( event ) {
        dispatchEvent(event);
        event->release();
    }
}


//====================================================================================================
// AppleEmbeddedHIDEventService::dispatchGyroEvent
//====================================================================================================
void AppleEmbeddedHIDEventService::dispatchGyroEvent(AbsoluteTime timestamp, IOFixed x, IOFixed y, IOFixed z, IOHIDMotionType type, IOHIDMotionPath subType, UInt32 sequence, IOOptionBits options)
{
    IOHIDEvent * event = IOHIDEvent::gyroEvent(timestamp, x, y, z, type, subType, sequence, options);
    
    if ( event ) {
        dispatchEvent(event);
        event->release();
    }
}

//====================================================================================================
// AppleEmbeddedHIDEventService::dispatchCompassEvent
//====================================================================================================
void AppleEmbeddedHIDEventService::dispatchCompassEvent(AbsoluteTime timestamp, IOFixed x, IOFixed y, IOFixed z, IOHIDMotionType type, IOHIDMotionPath subType, UInt32 sequence, IOOptionBits options)
{
    IOHIDEvent * event = IOHIDEvent::compassEvent(timestamp, x, y, z, type, subType, sequence, options);
    
    if ( event ) {
        dispatchEvent(event);
        event->release();
    }
}

//====================================================================================================
// AppleEmbeddedHIDEventService::dispatchProximityEvent
//====================================================================================================
void AppleEmbeddedHIDEventService::dispatchProximityEvent(AbsoluteTime timestamp, IOHIDProximityDetectionMask mask, UInt32 level, IOOptionBits options)
{
    IOHIDEvent * event = IOHIDEvent::proximityEvent(timestamp, mask, level, options);
    
    if ( event ) {
        dispatchEvent(event);
        event->release();
    }
}

//====================================================================================================
// AppleEmbeddedHIDEventService::dispatchAmbientLightSensorEvent
//====================================================================================================
void AppleEmbeddedHIDEventService::dispatchAmbientLightSensorEvent(AbsoluteTime timestamp, UInt32 level, UInt32 channel0, UInt32 channel1, UInt32 channel2, UInt32 channel3, IOOptionBits options)
{
    IOHIDEvent * event = IOHIDEvent::ambientLightSensorEvent(timestamp, level, channel0, channel1, channel2, channel3, options);
    
    if ( event ) {
        dispatchEvent(event);
        event->release();
    }
}

//====================================================================================================
// AppleEmbeddedHIDEventService::dispatchAmbientLightSensorEvent
//====================================================================================================

void AppleEmbeddedHIDEventService::dispatchAmbientLightSensorEvent(AbsoluteTime timestamp, UInt32 level, IOHIDEventColorSpace colorSpace, IOHIDDouble colorComponent0, IOHIDDouble colorComponent1, IOHIDDouble colorComponent2, IOOptionBits options)
{
    IOHIDEvent * event = IOHIDEvent::ambientLightSensorEvent(timestamp, level, colorSpace, colorComponent0, colorComponent1, colorComponent2, options);
    
    if ( event ) {
        dispatchEvent(event);
        event->release();
    }
}


//====================================================================================================
// AppleEmbeddedHIDEventService::dispatchTemperatureEvent
//====================================================================================================
void AppleEmbeddedHIDEventService::dispatchTemperatureEvent(AbsoluteTime timestamp, IOFixed temperature, IOOptionBits options)
{
    IOHIDEvent * event = IOHIDEvent::temperatureEvent(timestamp, temperature, options);
    
    if ( event ) {
        dispatchEvent(event);
        event->release();
    }
}

//====================================================================================================
// AppleEmbeddedHIDEventService::dispatchPowerEvent
//====================================================================================================
void AppleEmbeddedHIDEventService::dispatchPowerEvent(AbsoluteTime timestamp, int64_t measurement, IOHIDPowerType powerType, IOHIDPowerSubType powerSubType, IOOptionBits options)
{
    IOHIDEvent * event = IOHIDEvent::powerEvent(timestamp, measurement, powerType, powerSubType, options);
    
    if ( event ) {
        dispatchEvent(event);
        event->release();
    }
}

//====================================================================================================
// AppleEmbeddedHIDEventService::dispatchVendorDefinedEvent
//====================================================================================================
void AppleEmbeddedHIDEventService::dispatchVendorDefinedEvent(AbsoluteTime timeStamp, UInt32 usagePage, UInt32 usage, UInt32 version, UInt8 * data, UInt32 length, IOOptionBits options)
{
    IOHIDEvent * event = IOHIDEvent::vendorDefinedEvent(timeStamp, usagePage, usage, version, data, length, options);
    
    if ( event ) {
        dispatchEvent(event);
        event->release();
    }
}

//====================================================================================================
// AppleEmbeddedHIDEventService::dispatchBiometricEvent
//====================================================================================================
void AppleEmbeddedHIDEventService::dispatchBiometricEvent(AbsoluteTime timeStamp, IOFixed level, IOHIDBiometricEventType eventType, IOOptionBits options)
{
    IOHIDEvent * event = IOHIDEvent::biometricEvent(timeStamp, level, eventType, options);
    
    if ( event ) {
        dispatchEvent(event);
        event->release();
    }
}

//====================================================================================================
// AppleEmbeddedHIDEventService::dispatchBiometricEvent
//====================================================================================================
/*
void AppleEmbeddedHIDEventService::dispatchBiometricEvent(AbsoluteTime timeStamp, IOFixed level, IOHIDBiometricEventType eventType, UInt32 usagePage, UInt32 usage, UInt8 tapCount, IOOptionBits options)
{
    IOHIDEvent * event = IOHIDEvent::biometricEvent(timeStamp, level, eventType, usagePage, usage, tapCount, options);
    
    if ( event ) {
        dispatchEvent(event);
        event->release();
    }
}
*/

//====================================================================================================
// AppleEmbeddedHIDEventService::dispatchAtmosphericPressureEvent
//====================================================================================================
void AppleEmbeddedHIDEventService::dispatchAtmosphericPressureEvent(AbsoluteTime timeStamp, IOFixed level, UInt32 sequence, IOOptionBits options)
{
    IOHIDEvent * event = IOHIDEvent::atmosphericPressureEvent(timeStamp, level, sequence, options);
    
    if ( event ) {
        dispatchEvent(event);
        event->release();
    }
}

//====================================================================================================
// AppleEmbeddedHIDEventService::getOrientation
//====================================================================================================
IOHIDOrientationType AppleEmbeddedHIDEventService::getOrientation()
{
    return 0;
}

//====================================================================================================
// AppleEmbeddedHIDEventService::getPlacement
//====================================================================================================
IOHIDPlacementType AppleEmbeddedHIDEventService::getPlacement()
{
    return 0;
}

//====================================================================================================
// AppleEmbeddedHIDEventService::getReportInterval
//====================================================================================================
UInt32 AppleEmbeddedHIDEventService::getReportInterval()
{
    return 0;
}

