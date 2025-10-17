/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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
#ifndef _IOKIT_IOAUDIOCONTROLUSERCLIENT_H
#define _IOKIT_IOAUDIOCONTROLUSERCLIENT_H

#include <AvailabilityMacros.h>
#include <IOKit/IOUserClient.h>

#ifndef IOAUDIOFAMILY_SELF_BUILD
#include <IOKit/audio/IOAudioTypes.h>
#else
#include "IOAudioTypes.h"
#endif

class IOAudioControl;

class IOAudioControlUserClient : public IOUserClient
{
    OSDeclareDefaultStructors(IOAudioControlUserClient)
    
protected:
    task_t 				clientTask;
    IOAudioControl *			audioControl;
    IOAudioNotificationMessage *	notificationMessage;

    virtual IOReturn clientClose( ) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;
    virtual IOReturn clientDied( ) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;

protected:
    struct ExpansionData { };
    
    ExpansionData *reserved;

public:
	virtual void sendChangeNotification(UInt32 notificationType ) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;
    // OSMetaClassDeclareReservedUsed(IOAudioControlUserClient, 1);
    virtual bool initWithAudioControl(IOAudioControl *control, task_t owningTask, void *securityID, UInt32 type, OSDictionary *properties ) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;

private:
    OSMetaClassDeclareReservedUsed(IOAudioControlUserClient, 0);
    OSMetaClassDeclareReservedUsed(IOAudioControlUserClient, 1);

    OSMetaClassDeclareReservedUnused(IOAudioControlUserClient, 2);
    OSMetaClassDeclareReservedUnused(IOAudioControlUserClient, 3);
    OSMetaClassDeclareReservedUnused(IOAudioControlUserClient, 4);
    OSMetaClassDeclareReservedUnused(IOAudioControlUserClient, 5);
    OSMetaClassDeclareReservedUnused(IOAudioControlUserClient, 6);
    OSMetaClassDeclareReservedUnused(IOAudioControlUserClient, 7);
    OSMetaClassDeclareReservedUnused(IOAudioControlUserClient, 8);
    OSMetaClassDeclareReservedUnused(IOAudioControlUserClient, 9);
    OSMetaClassDeclareReservedUnused(IOAudioControlUserClient, 10);
    OSMetaClassDeclareReservedUnused(IOAudioControlUserClient, 11);
    OSMetaClassDeclareReservedUnused(IOAudioControlUserClient, 12);
    OSMetaClassDeclareReservedUnused(IOAudioControlUserClient, 13);
    OSMetaClassDeclareReservedUnused(IOAudioControlUserClient, 14);
    OSMetaClassDeclareReservedUnused(IOAudioControlUserClient, 15);

public:
    static IOAudioControlUserClient *withAudioControl(IOAudioControl *control, task_t clientTask, void *securityID, UInt32 type ) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;
    static IOAudioControlUserClient *withAudioControl(IOAudioControl *control, task_t clientTask, void *securityID, UInt32 type, OSDictionary *properties ) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;

    virtual bool initWithAudioControl(IOAudioControl *control, task_t owningTask, void *securityID, UInt32 type ) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;
	
    virtual void free( ) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;

    virtual IOReturn registerNotificationPort(mach_port_t port, UInt32 type, UInt32 refCon ) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;

    virtual void sendValueChangeNotification( ) AVAILABLE_MAC_OS_X_VERSION_10_4_AND_LATER_BUT_DEPRECATED_IN_MAC_OS_X_VERSION_10_10;
};

#endif /* _IOKIT_IOAUDIOCONTROLUSERCLIENT_H */
