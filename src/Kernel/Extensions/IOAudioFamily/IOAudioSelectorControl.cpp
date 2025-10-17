/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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
#include "IOAudioSelectorControl.h"
#include "IOAudioTypes.h"
#include "IOAudioDefines.h"

#include <libkern/c++/OSString.h>
#include <libkern/c++/OSArray.h>
#include <libkern/c++/OSDictionary.h>

#define super IOAudioControl

OSDefineMetaClassAndStructors(IOAudioSelectorControl, IOAudioControl)
OSMetaClassDefineReservedUsed(IOAudioSelectorControl, 0);
OSMetaClassDefineReservedUsed(IOAudioSelectorControl, 1);
OSMetaClassDefineReservedUsed(IOAudioSelectorControl, 2);
OSMetaClassDefineReservedUsed(IOAudioSelectorControl, 3);										// <rdar://8202424>

OSMetaClassDefineReservedUnused(IOAudioSelectorControl, 4);										// <rdar://8202424>
OSMetaClassDefineReservedUnused(IOAudioSelectorControl, 5);
OSMetaClassDefineReservedUnused(IOAudioSelectorControl, 6);
OSMetaClassDefineReservedUnused(IOAudioSelectorControl, 7);
OSMetaClassDefineReservedUnused(IOAudioSelectorControl, 8);
OSMetaClassDefineReservedUnused(IOAudioSelectorControl, 9);
OSMetaClassDefineReservedUnused(IOAudioSelectorControl, 10);
OSMetaClassDefineReservedUnused(IOAudioSelectorControl, 11);
OSMetaClassDefineReservedUnused(IOAudioSelectorControl, 12);
OSMetaClassDefineReservedUnused(IOAudioSelectorControl, 13);
OSMetaClassDefineReservedUnused(IOAudioSelectorControl, 14);
OSMetaClassDefineReservedUnused(IOAudioSelectorControl, 15);

// New code
IOAudioSelectorControl *IOAudioSelectorControl::createOutputClockSelector(SInt32 initialValue,
                                                                    UInt32 channelID,
																	UInt32 clockSource,
                                                                    const char *channelName,
                                                                    UInt32 cntrlID)
{
	IOAudioSelectorControl *clockControl;

	if ((clockControl = create (initialValue, 
                    channelID, 
                    channelName, 
                    cntrlID, 
                    kIOAudioSelectorControlSubTypeClockSource, 
                    kIOAudioControlUsageOutput))) {
        clockControl->setProperty(kIOAudioSelectorControlClockSourceKey, clockSource);
	}

	return clockControl;
}

IOAudioSelectorControl *IOAudioSelectorControl::createInputClockSelector(SInt32 initialValue,
                                                                    UInt32 channelID,
																	UInt32 clockSource,
                                                                    const char *channelName,
                                                                    UInt32 cntrlID)
{
	IOAudioSelectorControl *clockControl;

	if ((clockControl = create (initialValue, 
                    channelID, 
                    channelName, 
                    cntrlID, 
                    kIOAudioSelectorControlSubTypeClockSource, 
                    kIOAudioControlUsageInput))) {
        clockControl->setProperty(kIOAudioSelectorControlClockSourceKey, clockSource);
	}

	return clockControl;
}

IOAudioSelectorControl *IOAudioSelectorControl::createOutputSelector(SInt32 initialValue,
                                                                    UInt32 channelID,
                                                                    const char *channelName,
                                                                    UInt32 cntrlID)
{
    return create(initialValue, 
                    channelID, 
                    channelName, 
                    cntrlID, 
                    kIOAudioSelectorControlSubTypeOutput, 
                    kIOAudioControlUsageOutput);
}

IOReturn IOAudioSelectorControl::removeAvailableSelection(SInt32 selectionValue)
{
    OSCollectionIterator *iterator;
	OSArray *newSelections;
	OSArray *oldAvailableSelections;
    IOReturn result = kIOReturnNotFound;

    assert(availableSelections);

	oldAvailableSelections = availableSelections;
	newSelections = OSArray::withArray(availableSelections);
	if (!newSelections)
		return kIOReturnNoMemory;

    iterator = OSCollectionIterator::withCollection(newSelections);
    if (iterator) {
        OSDictionary *	selection;
		UInt32			index;

		index = 0;
        while ( (selection = (OSDictionary *)iterator->getNextObject()) ) {
            OSNumber *	sValue;

            sValue = (OSNumber *)selection->getObject(kIOAudioSelectorControlSelectionValueKey);

            if (sValue && ((SInt32)sValue->unsigned32BitValue() == selectionValue)) {
				// Remove the selected dictionary from the array
				newSelections->removeObject(index);
				result = kIOReturnSuccess;
                break;
            }
			index++;
        }
		availableSelections = newSelections;
        setProperty(kIOAudioSelectorControlAvailableSelectionsKey, availableSelections);
		oldAvailableSelections->release();

        iterator->release();
    }

	if (kIOReturnSuccess == result) {
		sendChangeNotification(kIOAudioControlRangeChangeNotification);
	}

    return result;
}

IOReturn IOAudioSelectorControl::replaceAvailableSelection(SInt32 selectionValue, const char *selectionDescription)
{
    IOReturn result = kIOReturnBadArgument;
    
    if (selectionDescription != NULL) {
        OSString *selDesc;
        
        selDesc = OSString::withCString(selectionDescription);
        if (selDesc) {
            result = replaceAvailableSelection(selectionValue, selDesc);
        } else {
            result = kIOReturnNoMemory;
        }
    }
    
    return result;
}

IOReturn IOAudioSelectorControl::replaceAvailableSelection(SInt32 selectionValue, OSString *selectionDescription)
{
    OSCollectionIterator *iterator;
	OSArray *newSelections;
	OSArray *oldAvailableSelections;
    IOReturn result = kIOReturnSuccess;
    
    assert(availableSelections);

	oldAvailableSelections = availableSelections;
	newSelections = OSArray::withArray(availableSelections);
	if (!newSelections)
		return kIOReturnNoMemory;

    iterator = OSCollectionIterator::withCollection(newSelections);
    if (iterator) {
        OSDictionary *	selection;
		UInt32			index;

		index = 0;
        while ( (selection = (OSDictionary *)iterator->getNextObject() )) {
            OSNumber *	sValue;

            sValue = (OSNumber *)selection->getObject(kIOAudioSelectorControlSelectionValueKey);

            if (sValue && ((SInt32)sValue->unsigned32BitValue() == selectionValue)) {
				// Replace the selected dictionary in the array
				newSelections->replaceObject(index, selectionDescription);
				result = kIOReturnSuccess;
                break;
            }
			index++;
        }
		availableSelections = newSelections;
        setProperty(kIOAudioSelectorControlAvailableSelectionsKey, availableSelections);
		oldAvailableSelections->release();

        iterator->release();
    }

	if (kIOReturnSuccess == result) {
		sendChangeNotification(kIOAudioControlRangeChangeNotification);
	}

    return result;
}

// Original code...
IOAudioSelectorControl *IOAudioSelectorControl::create(SInt32 initialValue,
                                                        UInt32 channelID,
                                                        const char *channelName,
                                                        UInt32 cntrlID,
                                                        UInt32 subType,
                                                        UInt32 usage)
{
    IOAudioSelectorControl *control;
    
    control = new IOAudioSelectorControl;
    
    if (control) {
        if (!control->init(initialValue,
                            channelID,
                            channelName,
                            cntrlID,
                            subType,
                            usage)) {
            control->release();
            control = NULL;
        }
    }
    
    return control;
}
                                            
IOAudioSelectorControl *IOAudioSelectorControl::createInputSelector(SInt32 initialValue,
                                                                    UInt32 channelID,
                                                                    const char *channelName,
                                                                    UInt32 cntrlID)
{
    return create(initialValue, 
                    channelID, 
                    channelName, 
                    cntrlID, 
                    kIOAudioSelectorControlSubTypeInput, 
                    kIOAudioControlUsageInput);
}

bool IOAudioSelectorControl::init(SInt32 initialValue,
                                    UInt32 channelID,
                                    const char *channelName,
                                    UInt32 cntrlID,
                                    UInt32 subType,
                                    UInt32 usage,
                                    OSDictionary *properties)
{
    bool result = false;
    OSNumber *number;
    
    number = OSNumber::withNumber(initialValue, sizeof(SInt32)*8);
    
    if (number) {
        result = super::init(kIOAudioControlTypeSelector, 
                        number,
                        channelID,
                        channelName,
                        cntrlID,
                        subType,
                        usage,
                        properties);
                        
        number->release();
    }
    
    if (result) {
        availableSelections = OSArray::withCapacity(2);
        setProperty(kIOAudioSelectorControlAvailableSelectionsKey, availableSelections);
    }
    
    return result;
}

void IOAudioSelectorControl::free()
{
    if (availableSelections) {
        availableSelections->release();
        availableSelections = NULL;
    }
    
    super::free();
}

IOReturn IOAudioSelectorControl::addAvailableSelection(SInt32 selectionValue, const char *selectionDescription)
{	
    IOReturn result = kIOReturnBadArgument;
    
    if (selectionDescription != NULL) {
        OSString *selDesc;
        
        selDesc = OSString::withCString(selectionDescription);
        if (selDesc) {
            result = addAvailableSelection(selectionValue, selDesc);
        } else {
            result = kIOReturnNoMemory;
        }
    }
    
    return result;
}

IOReturn IOAudioSelectorControl::addAvailableSelection(SInt32 selectionValue, OSString *selectionDescription)
{
	OSArray *newSelections;
	OSArray *oldAvailableSelections;
    IOReturn result = kIOReturnSuccess;
    
	oldAvailableSelections = availableSelections;
	newSelections = OSArray::withArray(availableSelections);
	if (!newSelections)
		return kIOReturnNoMemory;

    if (selectionDescription == NULL) {
        result = kIOReturnBadArgument;
    } else {
        if (valueExists(selectionValue)) {
            result = kIOReturnError;
        } else {
            OSDictionary *newSelection;
            
            newSelection = OSDictionary::withCapacity(2);
            
            if (newSelection) {
                OSNumber *number;

                number = OSNumber::withNumber(selectionValue, sizeof(SInt32)*8);
                
                if (number) {
                    newSelection->setObject(kIOAudioSelectorControlSelectionValueKey, number);
                    newSelection->setObject(kIOAudioSelectorControlSelectionDescriptionKey, selectionDescription);
                    newSelections->setObject(newSelection);

                    number->release();
                } else {
                    result = kIOReturnError;
                }
				availableSelections = newSelections;
				setProperty(kIOAudioSelectorControlAvailableSelectionsKey, availableSelections);
				oldAvailableSelections->release();
                
                newSelection->release();
            } else {
                result = kIOReturnError;
            }
        }
    }
    
	if (kIOReturnSuccess == result) {
		sendChangeNotification(kIOAudioControlRangeChangeNotification);
	}

    return result;
}

// <rdar://8202424>
IOReturn IOAudioSelectorControl::addAvailableSelection(SInt32 selectionValue, OSString *selectionDescription, const char* tagName, OSObject* tag)
{
	OSArray *newSelections;
	OSArray *oldAvailableSelections;
    IOReturn result = kIOReturnSuccess;
    
	oldAvailableSelections = availableSelections;
	newSelections = OSArray::withArray(availableSelections);
	if (!newSelections)
		return kIOReturnNoMemory;

    if (selectionDescription == NULL) {
        result = kIOReturnBadArgument;
    } else {
        if (valueExists(selectionValue)) {
            result = kIOReturnError;
        } else {
            OSDictionary *newSelection;
            
            newSelection = OSDictionary::withCapacity(2);
            
            if (newSelection) {
                OSNumber *number;
				
                number = OSNumber::withNumber(selectionValue, sizeof(SInt32)*8);
                
                if (number) {
                    newSelection->setObject(kIOAudioSelectorControlSelectionValueKey, number);
                    newSelection->setObject(kIOAudioSelectorControlSelectionDescriptionKey, selectionDescription);
                    newSelections->setObject(newSelection);
					
                    number->release();
                } else {
                    result = kIOReturnError;
                }
				
				if ( tagName && tag ) {
					newSelection->setObject(tagName, tag);
				}
				
				availableSelections = newSelections;
				setProperty(kIOAudioSelectorControlAvailableSelectionsKey, availableSelections);
				oldAvailableSelections->release();
                
                newSelection->release();
            } else {
                result = kIOReturnError;
            }
        }
    }
    
	if (kIOReturnSuccess == result) {
		sendChangeNotification(kIOAudioControlRangeChangeNotification);
	}
	
    return result;
}

bool IOAudioSelectorControl::valueExists(SInt32 selectionValue)
{
    bool found = false;
    OSCollectionIterator *iterator;

    assert(availableSelections);
    
    iterator = OSCollectionIterator::withCollection(availableSelections);
    if (iterator) {
        OSDictionary *selection;
        
        while ( (selection = (OSDictionary *)iterator->getNextObject()) ) {
            OSNumber *sValue;
            
            sValue = (OSNumber *)selection->getObject(kIOAudioSelectorControlSelectionValueKey);
            
            if (sValue && ((SInt32)sValue->unsigned32BitValue() == selectionValue)) {
                found = true;
                break;
            }
        }
        
        iterator->release();
    }
    
    return found;
}

IOReturn IOAudioSelectorControl::validateValue(OSObject *newValue)
{
    IOReturn result = kIOReturnBadArgument;
    OSNumber *number;
    
    number = OSDynamicCast(OSNumber, newValue);

    if (number) {
        result = super::validateValue(newValue);
        
        if (result == kIOReturnSuccess) {
            if (valueExists((SInt32)number->unsigned32BitValue())) {
                result = kIOReturnSuccess;
            } else {
                result = kIOReturnNotFound;
            }
        }
    }
    
    return result;
}

