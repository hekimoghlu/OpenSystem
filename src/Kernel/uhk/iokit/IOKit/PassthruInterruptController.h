/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 16, 2022.
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
#pragma once

extern "C" {
#include <mach/semaphore.h>
};

#include <IOKit/IOInterruptController.h>

/*!
 * @class       PassthruInterruptController
 * @abstract    Trivial IOInterruptController class that passes all IRQs through to a
 *              "child" driver.
 * @discussion  Waits for a "child" driver (typically loaded in a kext) to register itself,
 *              then passes the child driver's IOService pointer back via
 *              waitForChildController() so that XNU can operate on it directly.
 */
class PassthruInterruptController : public IOInterruptController
{
	OSDeclareDefaultStructors(PassthruInterruptController);

public:
	virtual bool     init(void) APPLE_KEXT_OVERRIDE;

	virtual void     *waitForChildController(void);

	virtual void     setCPUInterruptProperties(IOService *service) APPLE_KEXT_OVERRIDE;

	virtual IOReturn registerInterrupt(IOService *nub, int source,
	    void *target,
	    IOInterruptHandler handler,
	    void *refCon) APPLE_KEXT_OVERRIDE;

	virtual IOReturn getInterruptType(IOService *nub, int source,
	    int *interruptType) APPLE_KEXT_OVERRIDE;

	virtual IOReturn enableInterrupt(IOService *nub, int source) APPLE_KEXT_OVERRIDE;
	virtual IOReturn disableInterrupt(IOService *nub, int source) APPLE_KEXT_OVERRIDE;
	virtual IOReturn causeInterrupt(IOService *nub, int source) APPLE_KEXT_OVERRIDE;

	virtual IOReturn handleInterrupt(void *refCon, IOService *nub,
	    int source) APPLE_KEXT_OVERRIDE;

	virtual void externalInterrupt(void);

protected:
	IOInterruptHandler child_handler;
	void               *child_target;
	void               *child_refCon;
	IOService          *child_nub;
	semaphore_t        child_sentinel;
};
