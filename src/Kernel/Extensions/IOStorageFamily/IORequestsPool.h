/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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
#ifndef _IOREQUESTSPOOL_H
#define _IOREQUESTSPOOL_H

#include <kern/queue.h>
#include <IOKit/IOLocks.h>

class IORequest;
class IOMapper;
class IORequestsPool {
public:
	IOReturn init(uint32_t numOfRequests, uint32_t maxIOSize, uint8_t numOfAddressBits, uint32_t allignment, IOMapper *mapper);
	void deinit();

	IORequest *getReguest();
	void putRequest(IORequest *request);
private:
	/* Protects the list */
	IOLock *fLock;

	/* List of free requests */
	queue_head_t fFreeRequests;
    
    /* waiters */
    uint32_t fNumOfWaiters;
};
#endif /* _IOREQUESTSPOOL_H */
