/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 21, 2024.
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

#include "TestIOServiceUserNotification.h"
#include <IOKit/IOService.h>
#include <IOKit/IOUserClient.h>
#include <IOKit/IOKitServer.h>
#include <kern/ipc_kobject.h>

#if DEVELOPMENT || DEBUG

OSDefineMetaClassAndStructors(TestIOServiceUserNotification, IOService);

OSDefineMetaClassAndStructors(TestIOServiceUserNotificationUserClient, IOUserClient);

bool
TestIOServiceUserNotification::start(IOService * provider)
{
	OSString * str = OSString::withCStringNoCopy("TestIOServiceUserNotificationUserClient");
	bool ret = IOService::start(provider);
	if (ret && str != NULL) {
		setProperty(gIOUserClientClassKey, str);
		registerService();
	}
	OSSafeReleaseNULL(str);
	return ret;
}


IOReturn
TestIOServiceUserNotificationUserClient::clientClose()
{
	if (!isInactive()) {
		terminate();
	}
	return kIOReturnSuccess;
}

IOReturn
TestIOServiceUserNotificationUserClient::externalMethod(uint32_t selector, IOExternalMethodArguments * args,
    IOExternalMethodDispatch * dispatch, OSObject * target, void * reference)
{
	registerService();
	return kIOReturnSuccess;
}

#endif /* DEVELOPMENT || DEBUG */
