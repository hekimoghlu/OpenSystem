/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 4, 2023.
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
#include "IOAudioLevelControl.h"
#include "IOAudioTypes.h"
#include "IOAudioDefines.h"
#include "IOAudioDebug.h"

#define super IOAudioControl

OSDefineMetaClassAndStructors(IOAudioLevelControl, IOAudioControl)

OSMetaClassDefineReservedUsed(IOAudioLevelControl, 0);

OSMetaClassDefineReservedUnused(IOAudioLevelControl, 1);
OSMetaClassDefineReservedUnused(IOAudioLevelControl, 2);
OSMetaClassDefineReservedUnused(IOAudioLevelControl, 3);
OSMetaClassDefineReservedUnused(IOAudioLevelControl, 4);
OSMetaClassDefineReservedUnused(IOAudioLevelControl, 5);
OSMetaClassDefineReservedUnused(IOAudioLevelControl, 6);
OSMetaClassDefineReservedUnused(IOAudioLevelControl, 7);
OSMetaClassDefineReservedUnused(IOAudioLevelControl, 8);
OSMetaClassDefineReservedUnused(IOAudioLevelControl, 9);
OSMetaClassDefineReservedUnused(IOAudioLevelControl, 10);
OSMetaClassDefineReservedUnused(IOAudioLevelControl, 11);
OSMetaClassDefineReservedUnused(IOAudioLevelControl, 12);
OSMetaClassDefineReservedUnused(IOAudioLevelControl, 13);
OSMetaClassDefineReservedUnused(IOAudioLevelControl, 14);
OSMetaClassDefineReservedUnused(IOAudioLevelControl, 15);

// New code added here
IOAudioLevelControl *IOAudioLevelControl::createPassThruVolumeControl (SInt32 initialValue,
                                                                SInt32 minValue,
                                                                SInt32 maxValue,
                                                                IOFixed minDB,
                                                                IOFixed maxDB,
                                                                UInt32 channelID,
                                                                const char *channelName,
                                                                UInt32 cntrlID)
{
    return IOAudioLevelControl::create(initialValue,
                                        minValue,
                                        maxValue,
                                        minDB,
                                        maxDB,
                                        channelID,
                                        channelName,
                                        cntrlID,
                                        kIOAudioLevelControlSubTypeVolume,
                                        kIOAudioControlUsagePassThru);
}

// OSMetaClassDefineReservedUnused(IOAudioLevelControl, 0);
void IOAudioLevelControl::setLinearScale(bool useLinearScale)
{
    setProperty(kIOAudioLevelControlUseLinearScale, useLinearScale, sizeof(bool)*8);
}

// Original code...
IOAudioLevelControl *IOAudioLevelControl::create(SInt32 initialValue,
                                                 SInt32 minValue,
                                                 SInt32 maxValue,
                                                 IOFixed minDB,
                                                 IOFixed maxDB,
                                                 UInt32 channelID,
                                                 const char *channelName,
                                                 UInt32 cntrlID,
                                                 UInt32 subType,
                                                 UInt32 usage)
{
    IOAudioLevelControl *control;

    control = new IOAudioLevelControl;

    if (control) {
        if (!control->init(initialValue,
                           minValue,
                           maxValue,
                           minDB,
                           maxDB,
                           channelID,
                           channelName,
                           cntrlID,
                           subType,
                           usage)) {
            control->release();
            control = 0;
        }
    }

    return control;
}

IOAudioLevelControl *IOAudioLevelControl::createVolumeControl(SInt32 initialValue,
                                                                SInt32 minValue,
                                                                SInt32 maxValue,
                                                                IOFixed minDB,
                                                                IOFixed maxDB,
                                                                UInt32 channelID,
                                                                const char *channelName,
                                                                UInt32 cntrlID,
                                                                UInt32 usage)
{
    return IOAudioLevelControl::create(initialValue,
                                        minValue,
                                        maxValue,
                                        minDB,
                                        maxDB,
                                        channelID,
                                        channelName,
                                        cntrlID,
                                        kIOAudioLevelControlSubTypeVolume,
                                        usage);
}

bool IOAudioLevelControl::init(SInt32 initialValue,
                               SInt32 _minValue,
                               SInt32 _maxValue,
                               IOFixed _minDB,
                               IOFixed _maxDB,
                               UInt32 channelID,
                               const char *channelName,
                               UInt32 cntrlID,
                               UInt32 subType,
                               UInt32 usage,
                               OSDictionary *properties)
{
    bool result = true;
    OSNumber *number;
    
    number = OSNumber::withNumber(initialValue, sizeof(SInt32)*8);
    
    if ((number == NULL) || !super::init(kIOAudioControlTypeLevel, number, channelID, channelName, cntrlID, subType, usage, properties)) {
        result = false;
        goto Done;
    }

    setMinValue(_minValue);
    setMaxValue(_maxValue);
    setMinDB(_minDB);
    setMaxDB(_maxDB);

Done:
    if (number) {
        number->release();
    }
            
    return result;
}

void IOAudioLevelControl::free()
{
    if (ranges) {
        ranges->release();
        ranges = NULL;
    }
    
    super::free();
}
                   
void IOAudioLevelControl::setMinValue(SInt32 newMinValue)
{
    minValue = newMinValue;
    setProperty(kIOAudioLevelControlMinValueKey, newMinValue, sizeof(SInt32)*8);
	sendChangeNotification(kIOAudioControlRangeChangeNotification);
}

SInt32 IOAudioLevelControl::getMinValue()
{
    return minValue;
}
    
void IOAudioLevelControl::setMaxValue(SInt32 newMaxValue)
{
    maxValue = newMaxValue;
    setProperty(kIOAudioLevelControlMaxValueKey, newMaxValue, sizeof(SInt32)*8);
	sendChangeNotification(kIOAudioControlRangeChangeNotification);
}

SInt32 IOAudioLevelControl::getMaxValue()
{
    return maxValue;
}
    
void IOAudioLevelControl::setMinDB(IOFixed newMinDB)
{
    minDB = newMinDB;
    setProperty(kIOAudioLevelControlMinDBKey, newMinDB, sizeof(IOFixed)*8);
	sendChangeNotification(kIOAudioControlRangeChangeNotification);
}

IOFixed IOAudioLevelControl::getMinDB()
{
    return minDB;
}
    
void IOAudioLevelControl::setMaxDB(IOFixed newMaxDB)
{
    setProperty(kIOAudioLevelControlMaxDBKey, newMaxDB, sizeof(IOFixed)*8);
	sendChangeNotification(kIOAudioControlRangeChangeNotification);
}

IOFixed IOAudioLevelControl::getMaxDB()
{
    return maxDB;
}

// Should only be done during init time - this is not thread safe
IOReturn IOAudioLevelControl::addRange(SInt32 minRangeValue, 
                                        SInt32 maxRangeValue, 
                                        IOFixed minRangeDB, 
                                        IOFixed maxRangeDB)
{
    IOReturn result = kIOReturnSuccess;
    
    // We should verify the new range doesn't overlap any others here
    
    if (ranges == NULL) {
        ranges = OSArray::withCapacity(1);
        if (ranges) {
            setProperty(kIOAudioLevelControlRangesKey, ranges);
        }
    }
    
    if (ranges) {
        OSDictionary *newRange;
		OSArray *newRanges;
		OSArray *oldRanges;
        
		oldRanges = ranges;
        newRanges = OSArray::withArray(ranges);
		if (!newRanges)
			return kIOReturnNoMemory;
		
        newRange = OSDictionary::withCapacity(4);
        if (newRange) {
            OSNumber *number;
            
            number = OSNumber::withNumber(minRangeValue, sizeof(SInt32)*8);
            newRange->setObject(kIOAudioLevelControlMinValueKey, number);
            number->release();
            
            number = OSNumber::withNumber(maxRangeValue, sizeof(SInt32)*8);
            newRange->setObject(kIOAudioLevelControlMaxValueKey, number);
            number->release();
            
            number = OSNumber::withNumber(minRangeDB, sizeof(IOFixed)*8);
            newRange->setObject(kIOAudioLevelControlMinDBKey, number);
            number->release();
            
            number = OSNumber::withNumber(maxRangeDB, sizeof(IOFixed)*8);
            newRange->setObject(kIOAudioLevelControlMaxDBKey, number);
            number->release();
            
            newRanges->setObject(newRange);
            setProperty(kIOAudioLevelControlRangesKey, newRanges);
			ranges = newRanges;
			oldRanges->release();
            
            newRange->release();
        } else {
            result = kIOReturnError;
        }
    } else {
        result = kIOReturnError;
    }
    
    return result;
}

IOReturn IOAudioLevelControl::addNegativeInfinity(SInt32 negativeInfinityValue)
{
    return addRange(negativeInfinityValue, negativeInfinityValue, kIOAudioLevelControlNegativeInfinity, kIOAudioLevelControlNegativeInfinity);
}

IOReturn IOAudioLevelControl::validateValue(OSObject *newValue)
{
    IOReturn result = kIOReturnBadArgument;
    OSNumber *number;
    
    number = OSDynamicCast(OSNumber, newValue);
	
	audioDebugIOLog(3, "+ IOAudioLevelControl::validateValue[%p] (%p)\n", this, newValue);
   
    if (number) {
        SInt32 newIntValue;
        
        newIntValue = (SInt32)number->unsigned32BitValue();

		audioDebugIOLog(3, "  IOAudioLevelControl::validateValue[%p] - newIntValue = %ld, min = %ld, max = %ld\n", this, (long int)newIntValue, (long int)minValue, (long int)maxValue);
        
        if ((newIntValue >= minValue) && (newIntValue <= maxValue)) {
            result = kIOReturnSuccess;
        } else {
            result = kIOReturnError;
        }
    }
    
	audioDebugIOLog(3, "- IOAudioLevelControl::validateValue[%p] (%p) returns 0x%lX\n", this, newValue, (long unsigned int)result );
    return result;
}


