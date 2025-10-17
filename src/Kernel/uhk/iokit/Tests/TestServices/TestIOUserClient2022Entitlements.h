/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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

#ifndef _IOKIT_TESTIOUSERCLIENT2022ENTITLEMENTS_H_
#define _IOKIT_TESTIOUSERCLIENT2022ENTITLEMENTS_H_

#include <IOKit/IOService.h>
#include <IOKit/IOUserClient.h>

#if (DEVELOPMENT || DEBUG)

class TestIOUserClient2022Entitlements : public IOService {
	OSDeclareDefaultStructors(TestIOUserClient2022Entitlements);

public:
	virtual bool start(IOService *provider) override;
};

class TestIOUserClient2022EntitlementsUserClient : public IOUserClient2022 {
	OSDeclareDefaultStructors(TestIOUserClient2022EntitlementsUserClient);


public:
	virtual bool start(IOService * provider) override;
	virtual IOReturn clientClose() override;
	IOReturn externalMethod(uint32_t selector, IOExternalMethodArgumentsOpaque * args) override;
	static IOReturn        extBasicMethod(OSObject * target, void * reference, IOExternalMethodArguments * arguments);
	static IOReturn        extPerSelectorCheck(OSObject * target, void * reference, IOExternalMethodArguments * arguments);
};

#endif /* (DEVELOPMENT || DEBUG) */

#endif /* _IOKIT_TESTIOUSERCLIENT2022ENTITLEMENTS_H_ */
