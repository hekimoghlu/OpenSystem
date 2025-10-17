/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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
#define IOKIT_ENABLE_SHARED_PTR

#include <IOKit/IODMAEventSource.h>
#include <IOKit/IOService.h>

#include "IOKitKernelInternal.h"


#define super IOEventSource
OSDefineMetaClassAndStructors(IODMAEventSource, IOEventSource);

bool
IODMAEventSource::init(OSObject *inOwner,
    IOService *inProvider,
    Action inCompletion,
    Action inNotification,
    UInt32 inDMAIndex)
{
	IOReturn result;

	if (!super::init(inOwner)) {
		return false;
	}

	if (inProvider == NULL) {
		return false;
	}

	dmaProvider = inProvider;
	dmaIndex = 0xFFFFFFFF;
	dmaCompletionAction = inCompletion;
	dmaNotificationAction = inNotification;

	dmaController.reset(IODMAController::getController(dmaProvider, inDMAIndex), OSRetain);
	if (dmaController == NULL) {
		return false;
	}

	result = dmaController->initDMAChannel(dmaProvider, this, &dmaIndex, inDMAIndex);
	if (result != kIOReturnSuccess) {
		return false;
	}

	queue_init(&dmaCommandsCompleted);
	dmaCommandsCompletedLock = IOSimpleLockAlloc();

	return true;
}

void
IODMAEventSource::free()
{
	if (dmaCommandsCompletedLock != NULL) {
		IOSimpleLockFree(dmaCommandsCompletedLock);
	}
	super::free();
}

OSSharedPtr<IODMAEventSource>
IODMAEventSource::dmaEventSource(OSObject *inOwner,
    IOService *inProvider,
    Action inCompletion,
    Action inNotification,
    UInt32 inDMAIndex)
{
	OSSharedPtr<IODMAEventSource> dmaES = OSMakeShared<IODMAEventSource>();

	if (dmaES && !dmaES->init(inOwner, inProvider, inCompletion, inNotification, inDMAIndex)) {
		return nullptr;
	}

	return dmaES;
}

IOReturn
IODMAEventSource::startDMACommand(IODMACommand *dmaCommand, IODirection direction, IOByteCount byteCount, IOByteCount byteOffset)
{
	IOReturn result;

	if ((dmaController == NULL) || (dmaIndex == 0xFFFFFFFF)) {
		return kIOReturnError;
	}

	if (dmaSynchBusy) {
		return kIOReturnBusy;
	}

	if (dmaCompletionAction == NULL) {
		dmaSynchBusy = true;
	}

	result = dmaController->startDMACommand(dmaIndex, dmaCommand, direction, byteCount, byteOffset);

	if (result != kIOReturnSuccess) {
		dmaSynchBusy = false;
		return result;
	}

	while (dmaSynchBusy) {
		sleepGate(&dmaSynchBusy, THREAD_UNINT);
	}

	return kIOReturnSuccess;
}

IOReturn
IODMAEventSource::stopDMACommand(bool flush, uint64_t timeout)
{
	if ((dmaController == NULL) || (dmaIndex == 0xFFFFFFFF)) {
		return kIOReturnError;
	}

	return dmaController->stopDMACommand(dmaIndex, flush, timeout);
}


IOReturn
IODMAEventSource::queryDMACommand(IODMACommand **dmaCommand, IOByteCount *transferCount, bool waitForIdle)
{
	if ((dmaController == NULL) || (dmaIndex == 0xFFFFFFFF)) {
		return kIOReturnError;
	}

	return dmaController->queryDMACommand(dmaIndex, dmaCommand, transferCount, waitForIdle);
}


IOByteCount
IODMAEventSource::getFIFODepth(IODirection direction)
{
	if ((dmaController == NULL) || (dmaIndex == 0xFFFFFFFF)) {
		return 0;
	}

	return dmaController->getFIFODepth(dmaIndex, direction);
}


IOReturn
IODMAEventSource::setFIFODepth(IOByteCount depth)
{
	if ((dmaController == NULL) || (dmaIndex == 0xFFFFFFFF)) {
		return kIOReturnError;
	}

	return dmaController->setFIFODepth(dmaIndex, depth);
}


IOByteCount
IODMAEventSource::validFIFODepth(IOByteCount depth, IODirection direction)
{
	if ((dmaController == NULL) || (dmaIndex == 0xFFFFFFFF)) {
		return kIOReturnError;
	}

	return dmaController->validFIFODepth(dmaIndex, depth, direction);
}


IOReturn
IODMAEventSource::setFrameSize(UInt8 byteCount)
{
	if ((dmaController == NULL) || (dmaIndex == 0xFFFFFFFF)) {
		return kIOReturnError;
	}

	return dmaController->setFrameSize(dmaIndex, byteCount);
}

// protected

bool
IODMAEventSource::checkForWork(void)
{
	IODMACommand     *dmaCommand = NULL;
	bool work, again;

	IOSimpleLockLock(dmaCommandsCompletedLock);
	work = !queue_empty(&dmaCommandsCompleted);
	if (work) {
		queue_remove_first(&dmaCommandsCompleted, dmaCommand, IODMACommand *, fCommandChain);
		again = !queue_empty(&dmaCommandsCompleted);
	} else {
		again = false;
	}
	IOSimpleLockUnlock(dmaCommandsCompletedLock);

	if (work) {
		(*dmaCompletionAction)(owner, this, dmaCommand, dmaCommand->reserved->fStatus, dmaCommand->reserved->fActualByteCount, dmaCommand->reserved->fTimeStamp);
	}

	return again;
}

void
IODMAEventSource::completeDMACommand(IODMACommand *dmaCommand)
{
	if (dmaCompletionAction != NULL) {
		IOSimpleLockLock(dmaCommandsCompletedLock);
		queue_enter(&dmaCommandsCompleted, dmaCommand, IODMACommand *, fCommandChain);
		IOSimpleLockUnlock(dmaCommandsCompletedLock);

		signalWorkAvailable();
	} else {
		dmaSynchBusy = false;
		wakeupGate(&dmaSynchBusy, true);
	}
}

void
IODMAEventSource::notifyDMACommand(IODMACommand *dmaCommand, IOReturn status, IOByteCount actualByteCount, AbsoluteTime timeStamp)
{
	dmaCommand->reserved->fStatus = status;
	dmaCommand->reserved->fActualByteCount = actualByteCount;
	dmaCommand->reserved->fTimeStamp = timeStamp;

	if (dmaNotificationAction != NULL) {
		(*dmaNotificationAction)(owner, this, dmaCommand, status, actualByteCount, timeStamp);
	}
}

IOReturn
IODMAEventSource::setDMAConfig(UInt32 newReqIndex)
{
	return dmaController->setDMAConfig(dmaIndex, dmaProvider, newReqIndex);
}

bool
IODMAEventSource::validDMAConfig(UInt32 newReqIndex)
{
	return dmaController->validDMAConfig(dmaIndex, dmaProvider, newReqIndex);
}
