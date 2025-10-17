/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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
#include "IOAudioToggleControl.h"
#include "IOAudioTypes.h"
#include "IOAudioDefines.h"

#define super IOAudioControl

OSDefineMetaClassAndStructors(IOAudioToggleControl, IOAudioControl)
OSMetaClassDefineReservedUnused(IOAudioToggleControl, 0);
OSMetaClassDefineReservedUnused(IOAudioToggleControl, 1);
OSMetaClassDefineReservedUnused(IOAudioToggleControl, 2);
OSMetaClassDefineReservedUnused(IOAudioToggleControl, 3);
OSMetaClassDefineReservedUnused(IOAudioToggleControl, 4);
OSMetaClassDefineReservedUnused(IOAudioToggleControl, 5);
OSMetaClassDefineReservedUnused(IOAudioToggleControl, 6);
OSMetaClassDefineReservedUnused(IOAudioToggleControl, 7);
OSMetaClassDefineReservedUnused(IOAudioToggleControl, 8);
OSMetaClassDefineReservedUnused(IOAudioToggleControl, 9);
OSMetaClassDefineReservedUnused(IOAudioToggleControl, 10);
OSMetaClassDefineReservedUnused(IOAudioToggleControl, 11);
OSMetaClassDefineReservedUnused(IOAudioToggleControl, 12);
OSMetaClassDefineReservedUnused(IOAudioToggleControl, 13);
OSMetaClassDefineReservedUnused(IOAudioToggleControl, 14);
OSMetaClassDefineReservedUnused(IOAudioToggleControl, 15);

// New code added here
IOAudioToggleControl *IOAudioToggleControl::createPassThruMuteControl (bool initialValue,
                                               UInt32 channelID,
                                               const char *channelName,
                                               UInt32 cntrlID)
{
    return create(initialValue, channelID, channelName, cntrlID, kIOAudioToggleControlSubTypeMute, kIOAudioControlUsagePassThru);
}

// Original code...
IOAudioToggleControl *IOAudioToggleControl::create(bool initialValue,
                                               UInt32 channelID,
                                               const char *channelName,
                                               UInt32 cntrlID,
                                               UInt32 subType,
                                               UInt32 usage)
{
    IOAudioToggleControl *control;

    control = new IOAudioToggleControl;

    if (control) {
        if (!control->init(initialValue, channelID, channelName, cntrlID, subType, usage)) {
             control->release();
             control = 0;
        }
    }

    return control;
}

IOAudioToggleControl *IOAudioToggleControl::createMuteControl(bool initialValue,
                                                                UInt32 channelID,
                                                                const char *channelName,
                                                                UInt32 cntrlID,
                                                                UInt32 usage)
{
    return create(initialValue, channelID, channelName, cntrlID, kIOAudioToggleControlSubTypeMute, usage);
}

bool IOAudioToggleControl::init(bool initialValue,
                              UInt32 channelID,
                              const char *channelName,
                              UInt32 cntrlID,
                              UInt32 subType,
                              UInt32 usage,
                              OSDictionary *properties)
{
    bool result = false;
    OSNumber *number;
    
    number = OSNumber::withNumber((initialValue == 0) ? 0 : 1, 8);
    
    if (number) {
    	result = super::init(kIOAudioControlTypeToggle, number, channelID, channelName, cntrlID, subType, usage, properties);
        
        number->release();
    }
    
    return result;
}

