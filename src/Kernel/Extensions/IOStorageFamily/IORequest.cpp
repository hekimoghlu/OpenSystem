/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 6, 2021.
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
#include "IORequest.h"

#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/IODMACommand.h>

IOReturn IORequest::init(uint32_t Index, uint32_t maxIOSize, uint8_t numOfAddressBits, uint32_t allignment, IOMapper *mapper)
{
	fIndex = Index;

	fSyncLock = IOLockAlloc();
	if (fSyncLock == NULL)
		goto FailedtoAllocLock;

	fDMACommand = IODMACommand::withSpecification(kIODMACommandOutputHost64,
						      numOfAddressBits,
						      maxIOSize,
						      IODMACommand::kMapped,
						      maxIOSize,
						      allignment,
						      mapper);
	if (fDMACommand == NULL)
		goto FailedToAllocDMACommand;

	return kIOReturnSuccess;

FailedToAllocDMACommand:
	IOLockFree(fSyncLock);
FailedtoAllocLock:
	return kIOReturnNoSpace;
}

void IORequest::deinit()
{
	OSSafeReleaseNULL(fDMACommand);
	IOLockFree(fSyncLock);
}

void IORequest::waitForCompletion()
{
	IOReturn retVal = kIOReturnSuccess;

	/* Wait for the request to be completed */
	IOLockLock(fSyncLock);

	if (!fSyncCompleted) {
		IOLockSleep(fSyncLock, this, THREAD_UNINT);
	}

	IOLockUnlock(fSyncLock);

}

void IORequest::signalCompleted(IOReturn status)
{
	/* Wake up sleeping thread */
	IOLockLock(fSyncLock);

	fSyncCompleted = true;
	fSyncStatus = status;
	IOLockWakeup(fSyncLock, this, true);

	IOLockUnlock(fSyncLock);
}

IOReturn IORequest::prepare(IOStorageCompletion *completion, IOMemoryDescriptor *ioBuffer, uint64_t *dmaAddr, uint64_t *dmaSize)
{
	UInt64 offset = 0;
	IODMACommand::Segment64	segment[32];
	UInt32 numOfSegments = 32;
	IOReturn retVal;

	fCompletion = *completion;

	/* The defualt is auto prepared */
	retVal = fDMACommand->setMemoryDescriptor(ioBuffer);
	if (retVal != kIOReturnSuccess)
		goto FailedToSet;

	retVal = fDMACommand->genIOVMSegments(&offset, &segment, &numOfSegments);
	if ((retVal != kIOReturnSuccess))
		goto FailedToGen;

	/* Memory must be mapped contiguosly to the IOVM */
	if (numOfSegments != 1) {
		retVal = kIOReturnInvalid;
		goto IncorrectMapping;
	}

	*dmaAddr = segment[0].fIOVMAddr;
	*dmaSize = segment[0].fLength;

	return kIOReturnSuccess;

IncorrectMapping:
FailedToGen:
	fDMACommand->clearMemoryDescriptor();
FailedToSet:
	return retVal;
}
void IORequest::reset()
{
	fDMACommand->clearMemoryDescriptor();
}

void IORequest::complete(IOReturn status, uint64_t bytesTransfered)
{
	/* The default is autocompleted */
	fDMACommand->clearMemoryDescriptor();

	/* Complete the IO */
	IOStorage::complete(&fCompletion, status, bytesTransfered);

}

IOReturn IORequest::prepareToWait()
{
	fSyncCompleted = false;
	
	return kIOReturnSuccess;
}

IODirection IORequest::getDirection()
{
	return fDMACommand->getMemoryDescriptor()->getDirection();
}
