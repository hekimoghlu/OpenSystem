/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 5, 2025.
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
extern "C" {
#include <mach/task.h>
#include <pexpert/pexpert.h>
};

#include <machine/machine_routines.h>
#include <IOKit/IOPlatformExpert.h>
#include <IOKit/IOService.h>
#include <IOKit/PassthruInterruptController.h>

#define super IOInterruptController
OSDefineMetaClassAndStructors(PassthruInterruptController, IOInterruptController);

bool
PassthruInterruptController::init(void)
{
	if (!super::init() ||
	    !this->setProperty(gPlatformInterruptControllerName, kOSBooleanTrue) ||
	    !this->attach(getPlatform())) {
		return false;
	}
	registerService();
	if (getPlatform()->registerInterruptController(gPlatformInterruptControllerName, this) != kIOReturnSuccess) {
		return false;
	}
	if (semaphore_create(kernel_task, &child_sentinel, SYNC_POLICY_FIFO, 0) != KERN_SUCCESS) {
		return false;
	}
	return true;
}

void
PassthruInterruptController::setCPUInterruptProperties(IOService *service)
{
	if ((service->getProperty(gIOInterruptControllersKey) != NULL) &&
	    (service->getProperty(gIOInterruptSpecifiersKey) != NULL)) {
		return;
	}

	long         zero = 0;
	OSArray *specifier = OSArray::withCapacity(1);
	OSData *tmpData = OSData::withValue(zero);
	specifier->setObject(tmpData);
	tmpData->release();
	service->setProperty(gIOInterruptSpecifiersKey, specifier);
	specifier->release();

	OSArray *controller = OSArray::withCapacity(1);
	controller->setObject(gPlatformInterruptControllerName);
	service->setProperty(gIOInterruptControllersKey, controller);
	controller->release();
}

IOReturn
PassthruInterruptController::registerInterrupt(IOService *nub,
    int source,
    void *target,
    IOInterruptHandler handler,
    void *refCon)
{
	child_handler = handler;
	child_nub = nub;
	child_target = target;
	child_refCon = refCon;

	// Wake up waitForChildController() to tell it that AIC is registered
	semaphore_signal(child_sentinel);
	return kIOReturnSuccess;
}

void *
PassthruInterruptController::waitForChildController(void)
{
	// Block if child controller isn't registered yet.  Assumes that this
	// is only called from one place.
	semaphore_wait(child_sentinel);

	// NOTE: Assumes that AppleInterruptController passes |this| as the target argument.
	return child_target;
}

IOReturn
PassthruInterruptController::getInterruptType(IOService */*nub*/,
    int /*source*/,
    int *interruptType)
{
	if (interruptType == NULL) {
		return kIOReturnBadArgument;
	}

	*interruptType = kIOInterruptTypeLevel;

	return kIOReturnSuccess;
}

IOReturn
PassthruInterruptController::enableInterrupt(IOService */*nub*/,
    int /*source*/)
{
	return kIOReturnSuccess;
}

IOReturn
PassthruInterruptController::disableInterrupt(IOService */*nub*/,
    int /*source*/)
{
	return kIOReturnSuccess;
}

IOReturn
PassthruInterruptController::causeInterrupt(IOService */*nub*/,
    int /*source*/)
{
	ml_cause_interrupt();
	return kIOReturnSuccess;
}

IOReturn
PassthruInterruptController::handleInterrupt(void */*refCon*/,
    IOService */*nub*/,
    int source)
{
	panic("handleInterrupt shouldn't be invoked directly");
}

void
PassthruInterruptController::externalInterrupt(void)
{
	child_handler(child_target, child_refCon, child_nub, 0);
}
