/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 6, 2022.
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

#ifndef _IOKIT_TESTIOCONNECTMAPMEMORYPORTLEAK45265408_H_
#define _IOKIT_TESTIOCONNECTMAPMEMORYPORTLEAK45265408_H_

#include <IOKit/IOService.h>
#include <IOKit/IOUserClient.h>
#include <IOKit/IOBufferMemoryDescriptor.h>

#if DEVELOPMENT || DEBUG

class TestIOConnectMapMemoryPortLeak45265408 : public IOService {
	OSDeclareDefaultStructors(TestIOConnectMapMemoryPortLeak45265408)

public:
	virtual bool start(IOService *provider) override;
};

class TestIOConnectMapMemoryPortLeak45265408UserClient : public IOUserClient {
	OSDeclareDefaultStructors(TestIOConnectMapMemoryPortLeak45265408UserClient);

public:
	// IOService overrides
	virtual bool start(IOService *provider) override;
	virtual void stop(IOService *provider) override;

	// IOUserClient overrides
	virtual IOReturn clientClose() override;
	virtual IOReturn clientMemoryForType(UInt32 type, IOOptionBits *flags, IOMemoryDescriptor **memory) override;
private:
	IOBufferMemoryDescriptor *  sharedMemory;
};

#endif /* DEVELOPMENT || DEBUG */

#endif /* _IOKIT_TESTIOCONNECTMAPMEMORYPORTLEAK45265408_H_ */
