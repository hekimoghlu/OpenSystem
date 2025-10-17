/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 12, 2024.
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
#ifndef _IOKIT_APPLEI386PLATFORM_H
#define _IOKIT_APPLEI386PLATFORM_H

#include <IOKit/IOPlatformExpert.h>
#include "AppleI386CPU.h"

class AppleI386PlatformExpert : public IOPlatformExpert {
	OSDeclareDefaultStructors(AppleI386PlatformExpert)

private:
	const OSSymbol *_interruptControllerName;
	AppleI386CPU *bootCPU;

	void setupPIC(IOService *nub);
	void setupBIOS(IOService *nub);

	static int handlePEHaltRestart(unsigned int type);

public:
	virtual bool init(OSDictionary *properties) APPLE_KEXT_OVERRIDE;
	virtual IOService *probe(IOService *provider, SInt32 *score) APPLE_KEXT_OVERRIDE;
	virtual bool start(IOService *provider) APPLE_KEXT_OVERRIDE;
	virtual bool configure(IOService *provider) APPLE_KEXT_OVERRIDE;
	virtual bool matchNubWithPropertyTable(IOService *nub, OSDictionary *table);
	virtual IOService *createNub(OSDictionary *from) APPLE_KEXT_OVERRIDE;
	virtual bool reserveSystemInterrupt(IOService *client, UInt32 vectorNumber, bool exclusive);
	virtual void releaseSystemInterrupt(IOService *client, UInt32 vectorNumber, bool exclusive);
	virtual bool setNubInterruptVectors(IOService *nub, const UInt32 vectors[], UInt32 vectorCount);
	virtual bool setNubInterruptVector(IOService *nub, UInt32 vector);
	virtual IOReturn callPlatformFunction(const OSSymbol *functionName, bool waitForFunction, void *param1, void *param2, void *param3, void *param4) APPLE_KEXT_OVERRIDE;
	virtual bool getModelName(char *name, int maxLengh) APPLE_KEXT_OVERRIDE;
	virtual bool getMachineName(char *name, int maxLength) APPLE_KEXT_OVERRIDE;
};

#endif /* ! _IOKIT_APPLEI386PLATFORM_H */
