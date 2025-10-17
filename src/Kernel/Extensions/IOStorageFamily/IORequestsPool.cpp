/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 19, 2023.
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
#include "IORequestsPool.h"
#include "IORequest.h"
#include <IOKit/IOTypes.h>
#include <IOKit/IOLib.h>

IOReturn IORequestsPool::init(uint32_t numOfRequests, uint32_t maxIOSize, uint8_t numOfAddressBits, uint32_t allignment, IOMapper *mapper)
{
	IOReturn retVal;
	IORequest *newRequest;

	/* Init requests list */
	fLock = IOLockAlloc();
	if (fLock == NULL)
		return kIOReturnNoSpace;

	queue_init(&fFreeRequests);

	for (uint32_t i = 0; i < numOfRequests; i++) {

		newRequest = IOMallocType(IORequest);
		if (newRequest == NULL) {
			retVal = kIOReturnNoSpace;
			goto FailedToAlloc;
		}

		retVal = newRequest->init(i, maxIOSize, numOfAddressBits, allignment, mapper);
		if ( retVal != kIOReturnSuccess)
			goto FailedToInit;

		queue_enter_first(&fFreeRequests, newRequest, class IORequest *, fRequests);
	}
	
    fNumOfWaiters = 0;

	return kIOReturnSuccess;

FailedToInit:
	/* The request isn't in the list and not initalized */
	IOFreeType(newRequest, IORequest);
FailedToAlloc:
	/* Remove all requests from the list and deinit them */
	deinit();
	return retVal;
}

void IORequestsPool::deinit()
{
	IORequest *request;

	while(!queue_empty(&fFreeRequests)) {
		queue_remove_first(&fFreeRequests, request, class IORequest *, fRequests);
		request->deinit();
		IOFreeType(request, IORequest);
	}
	
	IOLockFree(fLock);

}

IORequest *IORequestsPool::getReguest()
{
	IORequest *request = NULL;
	
	IOLockLock(fLock);

	/* If the pool is empty - wait for some request will be returned */
    while (queue_empty(&fFreeRequests)) {
        fNumOfWaiters++;
		IOLockSleep(fLock, &fFreeRequests, THREAD_UNINT);
        fNumOfWaiters--;
    }

	queue_remove_first(&fFreeRequests, request, class IORequest *, fRequests);

	IOLockUnlock(fLock);
	
	return request;
}

void IORequestsPool::putRequest(IORequest *request)
{

	IOLockLock(fLock);

	/* If some thread is waiting - wake it up */
	if (fNumOfWaiters > 0)
		IOLockWakeup(fLock, &fFreeRequests, true);

	queue_enter_first(&fFreeRequests, request, class IORequest *, fRequests);

	IOLockUnlock(fLock);
}
