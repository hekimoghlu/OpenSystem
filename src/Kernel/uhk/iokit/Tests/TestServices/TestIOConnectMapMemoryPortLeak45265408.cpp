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

#include "TestIOConnectMapMemoryPortLeak45265408.h"
#include <IOKit/IOKitKeys.h>

#if DEVELOPMENT || DEBUG

#define super IOService
OSDefineMetaClassAndStructors(TestIOConnectMapMemoryPortLeak45265408, IOService);

bool
TestIOConnectMapMemoryPortLeak45265408::start(IOService *provider)
{
	bool ret = super::start(provider);
	if (ret) {
		OSString * className = OSString::withCStringNoCopy("TestIOConnectMapMemoryPortLeak45265408UserClient");
		setProperty(gIOUserClientClassKey, className);
		OSSafeReleaseNULL(className);
		registerService();
	}
	return ret;
}

#undef super
#define super IOUserClient
OSDefineMetaClassAndStructors(TestIOConnectMapMemoryPortLeak45265408UserClient, IOUserClient);

bool
TestIOConnectMapMemoryPortLeak45265408UserClient::start(IOService *provider)
{
	bool ret = super::start(provider);
	if (ret) {
		setProperty(kIOUserClientSharedInstanceKey, kOSBooleanTrue);
		this->sharedMemory = IOBufferMemoryDescriptor::withOptions(kIOMemoryKernelUserShared, PAGE_SIZE);
		if (this->sharedMemory == NULL) {
			ret = false;
		}
	}

	return ret;
}

void
TestIOConnectMapMemoryPortLeak45265408UserClient::stop(IOService *provider)
{
	if (this->sharedMemory) {
		this->sharedMemory->release();
		this->sharedMemory = NULL;
	}
	super::stop(provider);
}

IOReturn
TestIOConnectMapMemoryPortLeak45265408UserClient::clientClose()
{
	if (!isInactive()) {
		terminate();
	}
	return kIOReturnSuccess;
}

IOReturn
TestIOConnectMapMemoryPortLeak45265408UserClient::clientMemoryForType(UInt32 type, IOOptionBits *flags, IOMemoryDescriptor **memory)
{
	*memory = this->sharedMemory;
	this->sharedMemory->retain();
	return kIOReturnSuccess;
}

#endif /* DEVELOPMENT || DEBUG */
