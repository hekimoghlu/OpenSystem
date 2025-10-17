/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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
#ifndef _IOKIT_IOAUDIOSELECTORCONTROL_H
#define _IOKIT_IOAUDIOSELECTORCONTROL_H

#include <AvailabilityMacros.h>

#ifndef IOAUDIOFAMILY_SELF_BUILD
#include <IOKit/audio/IOAudioControl.h>
#else
#include "IOAudioControl.h"
#endif

class OSString;
class OSArray;

class IOAudioSelectorControl : public IOAudioControl
{
    OSDeclareDefaultStructors(IOAudioSelectorControl)
    
protected:

    OSArray *availableSelections;

protected:
    struct ExpansionData { };
    
    ExpansionData *reserved;

public:
	static IOAudioSelectorControl *createOutputSelector(SInt32 initialValue,
															UInt32 channelID,
															const char *channelName = 0,
															UInt32 cntrlID = 0) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;
	static IOAudioSelectorControl *createOutputClockSelector(SInt32 initialValue,
                                                                    UInt32 channelID,
																	UInt32 clockSource,
                                                                    const char *channelName = 0,
                                                                    UInt32 cntrlID = 0) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;
	static IOAudioSelectorControl *createInputClockSelector(SInt32 initialValue,
                                                                    UInt32 channelID,
																	UInt32 clockSource,
                                                                    const char *channelName = 0,
                                                                    UInt32 cntrlID = 0) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;

    // OSMetaClassDeclareReservedUsed(IOAudioSelectorControl, 0);
	virtual IOReturn removeAvailableSelection(SInt32 selectionValue) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;
    // OSMetaClassDeclareReservedUsed(IOAudioSelectorControl, 1);
	virtual IOReturn replaceAvailableSelection(SInt32 selectionValue, const char *selectionDescription) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;
    // OSMetaClassDeclareReservedUsed(IOAudioSelectorControl, 2);
	virtual IOReturn replaceAvailableSelection(SInt32 selectionValue, OSString *selectionDescription) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;
    // OSMetaClassDeclareReservedUsed(IOAudioSelectorControl, 3);
	virtual IOReturn addAvailableSelection(SInt32 selectionValue, OSString *selectionDescription, const char* pszName, OSObject* tag) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;		// <rdar://8202424>

private:
    OSMetaClassDeclareReservedUsed(IOAudioSelectorControl, 0);
    OSMetaClassDeclareReservedUsed(IOAudioSelectorControl, 1);
    OSMetaClassDeclareReservedUsed(IOAudioSelectorControl, 2);
    OSMetaClassDeclareReservedUsed(IOAudioSelectorControl, 3);			// <rdar://8202424>

    OSMetaClassDeclareReservedUnused(IOAudioSelectorControl, 4);		// <rdar://8202424>

    OSMetaClassDeclareReservedUnused(IOAudioSelectorControl, 5);		// <rdar://8202424>
    OSMetaClassDeclareReservedUnused(IOAudioSelectorControl, 6);
    OSMetaClassDeclareReservedUnused(IOAudioSelectorControl, 7);
    OSMetaClassDeclareReservedUnused(IOAudioSelectorControl, 8);
    OSMetaClassDeclareReservedUnused(IOAudioSelectorControl, 9);
    OSMetaClassDeclareReservedUnused(IOAudioSelectorControl, 10);
    OSMetaClassDeclareReservedUnused(IOAudioSelectorControl, 11);
    OSMetaClassDeclareReservedUnused(IOAudioSelectorControl, 12);
    OSMetaClassDeclareReservedUnused(IOAudioSelectorControl, 13);
    OSMetaClassDeclareReservedUnused(IOAudioSelectorControl, 14);
    OSMetaClassDeclareReservedUnused(IOAudioSelectorControl, 15);

public:
    static IOAudioSelectorControl *create(SInt32 initialValue,
                                            UInt32 channelID,
                                            const char *channelName = 0,
                                            UInt32 cntrlID = 0,
                                            UInt32 subType = 0,
                                            UInt32 usage = 0) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;
                                            
    static IOAudioSelectorControl *createInputSelector(SInt32 initialValue,
                                                        UInt32 channelID,
                                                        const char *channelName = 0,
                                                        UInt32 cntrlID = 0) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;
                                                        
    virtual bool init(SInt32 initialValue,
                      UInt32 channelID,
                      const char *channelName = 0,
                      UInt32 cntrlID = 0,
                      UInt32 subType = 0,
                      UInt32 usage = 0,
                      OSDictionary *properties = 0) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;

    virtual void free() AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;

    virtual IOReturn addAvailableSelection(SInt32 selectionValue, const char *selectionDescription) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;
    virtual IOReturn addAvailableSelection(SInt32 selectionValue, OSString *selectionDescription) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;

    virtual bool valueExists(SInt32 selectorValue) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;

    virtual IOReturn validateValue(OSObject *newValue) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;

};

#endif /* _IOKIT_IOAUDIOSELECTORCONTROL_H */
