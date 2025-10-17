/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 25, 2023.
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
#ifndef _IOKIT_IOAUDIOTOGGLECONTROL_H
#define _IOKIT_IOAUDIOTOGGLECONTROL_H

#include <AvailabilityMacros.h>

#ifndef IOAUDIOFAMILY_SELF_BUILD
#include <IOKit/audio/IOAudioControl.h>
#else
#include "IOAudioControl.h"
#endif

/*!
 * @class IOAudioToggleControl
 */

class IOAudioToggleControl : public IOAudioControl
{
    OSDeclareDefaultStructors(IOAudioToggleControl)

protected:
    struct ExpansionData { };
    
    ExpansionData *reserved;

// New code added here
public:
    /*!
     * @function createPassThruMuteControl
     * @abstract Allocates a new pass through mute control with the given attributes
     * @param initialValue The initial value of the control
     * @param channelID The ID of the channel(s) that the control acts on.  Common IDs are located in IOAudioTypes.h.
     * @param channelName An optional name for the channel.  Common names are located in IOAudioPort.h.
     * @param cntrlID An optional ID for the control that can be used to uniquely identify controls
     * @result Returns a newly allocated and initialized mute IOAudioControl
     */
	static IOAudioToggleControl *createPassThruMuteControl (bool initialValue,
																UInt32 channelID,
																const char *channelName,
																UInt32 cntrlID) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;

private:
    OSMetaClassDeclareReservedUnused(IOAudioToggleControl, 0);
    OSMetaClassDeclareReservedUnused(IOAudioToggleControl, 1);
    OSMetaClassDeclareReservedUnused(IOAudioToggleControl, 2);
    OSMetaClassDeclareReservedUnused(IOAudioToggleControl, 3);
    OSMetaClassDeclareReservedUnused(IOAudioToggleControl, 4);
    OSMetaClassDeclareReservedUnused(IOAudioToggleControl, 5);
    OSMetaClassDeclareReservedUnused(IOAudioToggleControl, 6);
    OSMetaClassDeclareReservedUnused(IOAudioToggleControl, 7);
    OSMetaClassDeclareReservedUnused(IOAudioToggleControl, 8);
    OSMetaClassDeclareReservedUnused(IOAudioToggleControl, 9);
    OSMetaClassDeclareReservedUnused(IOAudioToggleControl, 10);
    OSMetaClassDeclareReservedUnused(IOAudioToggleControl, 11);
    OSMetaClassDeclareReservedUnused(IOAudioToggleControl, 12);
    OSMetaClassDeclareReservedUnused(IOAudioToggleControl, 13);
    OSMetaClassDeclareReservedUnused(IOAudioToggleControl, 14);
    OSMetaClassDeclareReservedUnused(IOAudioToggleControl, 15);

public:
    /*!
     * @function create
     * @abstract Allocates a new mute control with the given attributes
     * @param initialValue The initial value of the control
     * @param channelID The ID of the channel(s) that the control acts on.  Common IDs are located in IOAudioTypes.h.
     * @param channelName An optional name for the channel.  Common names are located in IOAudioPort.h.
     * @param cntrlID An optional ID for the control that can be used to uniquely identify controls
     * @result Returns a newly allocated and initialized mute IOAudioControl
     */
    static IOAudioToggleControl *create(bool initialValue,
                                        UInt32 channelID,
                                        const char *channelName = 0,
                                        UInt32 cntrlID = 0,
                                        UInt32 subType = 0,
                                        UInt32 usage = 0) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;
                                      
    static IOAudioToggleControl *createMuteControl(bool initialValue,
                                                    UInt32 channelID,
                                                    const char *channelName = 0,
                                                    UInt32 cntrlID = 0,
                                                    UInt32 usage = 0) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;

    /*!
     * @function init
     * @abstract Initializes a newly allocated IOAudioToggleControl with the given attributes
     * @param initialValue The initial value of the control
     * @param channelID The ID of the channel(s) that the control acts on.  Common IDs are located in IOAudioTypes.h.
     * @param channelName An optional name for the channel.  Common names are located in IOAudioPort.h.
     * @param cntrlID An optional ID for the control that can be used to uniquely identify controls
     * @result Returns truen on success
     */
    virtual bool init(bool initialValue,
                      UInt32 channelID, 
                      const char *channelName = 0,
                      UInt32 cntrlID = 0,
                      UInt32 subType = 0,
                      UInt32 usage = 0,
                      OSDictionary *properties = 0) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;

};

#endif /* _IOKIT_IOAUDIOTOGGLECONTROL_H */
