/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 20, 2022.
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

#include "TestIOUserClient2022Entitlements.h"
#include <IOKit/IOService.h>
#include <IOKit/IOUserClient.h>
#include <IOKit/IOKitServer.h>
#include <kern/ipc_kobject.h>

#if (DEVELOPMENT || DEBUG)

OSDefineMetaClassAndStructors(TestIOUserClient2022Entitlements, IOService);

OSDefineMetaClassAndStructors(TestIOUserClient2022EntitlementsUserClient, IOUserClient2022);

bool
TestIOUserClient2022Entitlements::start(IOService * provider)
{
	OSString * str = OSString::withCStringNoCopy("TestIOUserClient2022EntitlementsUserClient");
	bool ret = IOService::start(provider);
	if (ret && str != NULL) {
		setProperty(gIOUserClientClassKey, str);
		registerService();
	}
	OSSafeReleaseNULL(str);
	return ret;
}

bool
TestIOUserClient2022EntitlementsUserClient::start(IOService * provider)
{
	if (!IOUserClient2022::start(provider)) {
		return false;
	}
	setProperty(kIOUserClientDefaultLockingKey, kOSBooleanTrue);
	setProperty(kIOUserClientDefaultLockingSetPropertiesKey, kOSBooleanTrue);
	setProperty(kIOUserClientDefaultLockingSingleThreadExternalMethodKey, kOSBooleanTrue);

	setProperty(kIOUserClientEntitlementsKey, "com.apple.iokit.test-check-entitlement-open");

	return true;
}

IOReturn
TestIOUserClient2022EntitlementsUserClient::clientClose()
{
	terminate();
	return kIOReturnSuccess;
}

IOReturn
TestIOUserClient2022EntitlementsUserClient::extBasicMethod(OSObject * target, void * reference, IOExternalMethodArguments * arguments)
{
	return kIOReturnSuccess;
}

IOReturn
TestIOUserClient2022EntitlementsUserClient::extPerSelectorCheck(OSObject * target, void * reference, IOExternalMethodArguments * arguments)
{
	return kIOReturnSuccess;
}

IOReturn
TestIOUserClient2022EntitlementsUserClient::externalMethod(uint32_t selector, IOExternalMethodArgumentsOpaque * args)
{
	static const IOExternalMethodDispatch2022 dispatchArray[] = {
		[0] {
			.function                 = &TestIOUserClient2022EntitlementsUserClient::extBasicMethod,
			.checkScalarInputCount    = 0,
			.checkStructureInputSize  = 0,
			.checkScalarOutputCount   = 0,
			.checkStructureOutputSize = 0,
			.allowAsync               = false,
			.checkEntitlement         = NULL,
		},
		[1] {
			.function                 = &TestIOUserClient2022EntitlementsUserClient::extPerSelectorCheck,
			.checkScalarInputCount    = 0,
			.checkStructureInputSize  = 0,
			.checkScalarOutputCount   = 0,
			.checkStructureOutputSize = 0,
			.allowAsync               = false,
			.checkEntitlement         = "com.apple.iokit.test-check-entitlement-per-selector",
		},
	};

	return dispatchExternalMethod(selector, args, dispatchArray, sizeof(dispatchArray) / sizeof(dispatchArray[0]), this, NULL);
}

#endif /* (DEVELOPMENT || DEBUG) */
