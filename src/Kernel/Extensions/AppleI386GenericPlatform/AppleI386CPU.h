/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
#ifndef _IOKIT_APPLEI386CPU_H
#define _IOKIT_APPLEI386CPU_H

#include <IOKit/IOCPU.h>

class AppleI386CPU : public IOCPU {
	OSDeclareDefaultStructors(AppleI386CPU);

private:
	IOCPUInterruptController *cpuIC;
	bool startCommonCompleted;

public:
	virtual IOService *probe(IOService *provider, SInt32 *score) APPLE_KEXT_OVERRIDE;
	virtual bool start(IOService *provider) APPLE_KEXT_OVERRIDE;
	virtual void initCPU(bool boot) APPLE_KEXT_OVERRIDE;
	virtual void quiesceCPU(void) APPLE_KEXT_OVERRIDE;
	virtual kern_return_t startCPU(vm_offset_t start_paddr, vm_offset_t arg_paddr) APPLE_KEXT_OVERRIDE;
	virtual void haltCPU(void) APPLE_KEXT_OVERRIDE;
	virtual const OSSymbol *getCPUName(void) APPLE_KEXT_OVERRIDE;
	bool startCommon(void);
};

class AppleI386CPUInterruptController : public IOCPUInterruptController {
	OSDeclareDefaultStructors(AppleI386CPUInterruptController);

public:
	virtual IOReturn handleInterrupt(void *refCon, IOService *nub, int source) APPLE_KEXT_OVERRIDE;
};

#endif
