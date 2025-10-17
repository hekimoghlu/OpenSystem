/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 26, 2024.
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
#ifndef _IOREQUEST_H
#define _IOREQUEST_H

#include <IOKit/storage/IOStorage.h>
#include <kern/queue.h>
#include <IOKit/IOLocks.h>

class IOMemoryDescriptor;
class IODMACommand;
class IOMapper;

class IORequest {
public:
	IOReturn init(uint32_t index, uint32_t maxIOSize, uint8_t numOfAddressBits, uint32_t allignment, IOMapper *mapper);
	void deinit();

	IOReturn prepare(IOStorageCompletion *completion, IOMemoryDescriptor *ioBuffer, uint64_t *dmaAddr, uint64_t *dmaSize);
	void complete(IOReturn status, uint64_t bytesTransfered);
	void reset();

	void waitForCompletion();
	void signalCompleted(IOReturn status);
	IOReturn prepareToWait();

	IODirection getDirection();
	IOReturn getStatus() { return fSyncStatus; }

	uint32_t getIndex() { return fIndex; }

	void setIORequest(bool isIORequest) { fIsIORequest = isIORequest; }
	bool getIORequest() { return fIsIORequest; }
	
	void setBytesToTransfer(uint64_t bytestoTransfer) {fBytestoTransfer = bytestoTransfer; }
	uint64_t getBytesToTransfer() { return fBytestoTransfer; };
	queue_chain_t fRequests;

private:

	IODMACommand *fDMACommand;

	uint32_t fIndex;
	bool fIsIORequest;
	uint64_t fBytestoTransfer;
	IOStorageCompletion fCompletion;
	IOLock *fSyncLock;
	bool fSyncCompleted;
	IOReturn fSyncStatus;
};

#endif /* _IOREQUEST_H */
