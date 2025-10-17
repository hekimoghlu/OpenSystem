/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 21, 2024.
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
#include <CoreFoundation/CoreFoundation.h>

#include "UserEventAgentInterface.h"

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <asl.h>

#include "kcm.h"

typedef struct {
    UserEventAgentInterfaceStruct *agentInterface;
    CFUUIDRef factoryID;
    UInt32 refCount;

    CFRunLoopRef rl;

    CFNotificationCenterRef darwin;
    CFNotificationCenterRef distributed;

} GSSNotificationForwarder;

static void
GSSNotificationForwarderDelete(GSSNotificationForwarder *instance) {

    CFUUIDRef factoryID = instance->factoryID;

    asl_log(NULL, NULL, ASL_LEVEL_DEBUG, "UserEventAgentFactory: %s", __func__);

    /* XXX */

    if (instance->darwin)
	CFRelease(instance->darwin);
    if (instance->distributed)
	CFRelease(instance->distributed);
    if (instance->rl)
	CFRelease(instance->rl);

    if (factoryID) {
	CFPlugInRemoveInstanceForFactory(factoryID);
	CFRelease(factoryID);
    }
    free(instance);
}


static void
cc_changed(CFNotificationCenterRef center,
	   void *observer,
	   CFStringRef name,
	   const void *object,
	   CFDictionaryRef userInfo)
{
    GSSNotificationForwarder *instance = observer;

    CFNotificationCenterPostNotification(instance->distributed,
					 CFSTR(kCCAPICacheCollectionChangedNotification),
					 NULL,
					 NULL,
					 false);
}

static void
GSSNotificationForwarderInstall(void *pinstance)
{
    GSSNotificationForwarder *instance = pinstance;

    asl_log(NULL, NULL, ASL_LEVEL_DEBUG, "UserEventAgentFactory: %s", __func__);

    instance->darwin = CFNotificationCenterGetDarwinNotifyCenter();
    instance->distributed = CFNotificationCenterGetDistributedCenter();

    CFNotificationCenterAddObserver(instance->darwin,
				    instance,
				    cc_changed,
				    CFSTR(KRB5_KCM_NOTIFY_CACHE_CHANGED),
				    NULL,
				    CFNotificationSuspensionBehaviorHold);

}


static HRESULT
GSSNotificationForwarderQueryInterface(void *pinstance, REFIID iid, LPVOID *ppv)
{
    CFUUIDRef interfaceID = CFUUIDCreateFromUUIDBytes(NULL, iid);
    GSSNotificationForwarder *instance = pinstance;
        
    asl_log(NULL, NULL, ASL_LEVEL_DEBUG, "UserEventAgentFactory: %s", __func__);

    if (CFEqual(interfaceID, kUserEventAgentInterfaceID) || CFEqual(interfaceID, IUnknownUUID)) {
	instance->agentInterface->AddRef(instance);
	*ppv = instance;
	CFRelease(interfaceID);
	return S_OK;
    }

    *ppv = NULL;
    CFRelease(interfaceID);
    return E_NOINTERFACE;
}

static ULONG
GSSNotificationForwarderAddRef(void *pinstance) 
{
    GSSNotificationForwarder *instance = pinstance;
    return ++instance->refCount;
}

static ULONG
GSSNotificationForwarderRelease(void *pinstance) 
{
    GSSNotificationForwarder *instance = pinstance;
    if (instance->refCount == 1) {
	GSSNotificationForwarderDelete(instance);
	return 0;
    }
    return --instance->refCount;
}



static UserEventAgentInterfaceStruct UserEventAgentInterfaceFtbl = {
    NULL,
    GSSNotificationForwarderQueryInterface,
    GSSNotificationForwarderAddRef,
    GSSNotificationForwarderRelease,
    GSSNotificationForwarderInstall
}; 


static GSSNotificationForwarder *
GSSNotificationForwarderCreate(CFAllocatorRef allocator, CFUUIDRef factoryID)
{
    GSSNotificationForwarder *instance;

    asl_log(NULL, NULL, ASL_LEVEL_DEBUG, "UserEventAgentFactory: %s", __func__);

    instance = calloc(1, sizeof(*instance));
    if (instance == NULL)
	return NULL;

    instance->agentInterface = &UserEventAgentInterfaceFtbl;
    if (factoryID) {
	instance->factoryID = (CFUUIDRef)CFRetain(factoryID);
	CFPlugInAddInstanceForFactory(factoryID);
    }

    instance->rl = CFRunLoopGetCurrent();
    CFRetain(instance->rl);

    instance->refCount = 1;
    return instance;
}


void *
UserEventAgentFactory(CFAllocatorRef allocator, CFUUIDRef typeID);


void *
UserEventAgentFactory(CFAllocatorRef allocator, CFUUIDRef typeID)
{
    asl_log(NULL, NULL, ASL_LEVEL_DEBUG, "UserEventAgentFactory: %s", __func__);

    if (CFEqual(typeID, kUserEventAgentTypeID))
	return GSSNotificationForwarderCreate(allocator, kUserEventAgentFactoryID);

    return NULL;
}
