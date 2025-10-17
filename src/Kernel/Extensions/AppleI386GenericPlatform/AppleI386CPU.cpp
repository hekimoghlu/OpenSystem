/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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
#include "AppleI386CPU.h"

#undef super
#define super IOCPU

OSDefineMetaClassAndStructors(AppleI386CPU, IOCPU);

IOService *AppleI386CPU::probe(IOService *provider, SInt32 *score) {
	return this;
}

bool AppleI386CPU::startCommon() {
	if (startCommonCompleted) return true;

	cpuIC = new AppleI386CPUInterruptController;
	if (cpuIC == 0) return false;
	if (cpuIC->initCPUInterruptController(1) != kIOReturnSuccess) return false;

	cpuIC->attach(this);
	cpuIC->registerCPUInterruptController();

	setCPUState(kIOCPUStateUninitalized);
	initCPU(true);
	registerService();

	startCommonCompleted = true;
	return true;
}

bool AppleI386CPU::start(IOService *provider) {
	if (!super::start(provider)) return false;
	return startCommon();
}

void AppleI386CPU::initCPU(bool boot) {
	cpuIC->enableCPUInterrupt(this);
	setCPUState(kIOCPUStateRunning);
}

void AppleI386CPU::quiesceCPU() {
	// Not required.
}

kern_return_t AppleI386CPU::startCPU(vm_offset_t start_paddr, vm_offset_t arg_paddr) {
	// Not implemented.
	return KERN_FAILURE;
}

void AppleI386CPU::haltCPU() {
	// Not required.
}

const OSSymbol *AppleI386CPU::getCPUName() {
	return OSSymbol::withCStringNoCopy("Primary0");
}

#pragma mark -
#undef super
#define super IOCPUInterruptController

OSDefineMetaClassAndStructors(AppleI386CPUInterruptController, IOCPUInterruptController);

IOReturn AppleI386CPUInterruptController::handleInterrupt(void *refCon, IOService *nub, int source) {
	// Override the implementation in IOCPUInterruptController to
	// dispatch interrupts the old way. The source argument is ignored;
	// the first IOCPUInterruptController in the vector array is always used.

	IOInterruptVector *vector = &vectors[0];
	if (!vector->interruptRegistered) return kIOReturnInvalid;

	vector->handler(vector->target, refCon, vector->nub, source);
	return kIOReturnSuccess;
}
