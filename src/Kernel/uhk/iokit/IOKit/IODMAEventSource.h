/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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
#ifndef _IOKIT_IODMAEVENTSOURCE_H
#define _IOKIT_IODMAEVENTSOURCE_H

#include <libkern/c++/OSPtr.h>
#include <IOKit/IOService.h>
#include <IOKit/IODMACommand.h>
#include <IOKit/IODMAController.h>
#include <IOKit/IOEventSource.h>

class IODMAController;

class IODMAEventSource : public IOEventSource
{
	OSDeclareDefaultStructors(IODMAEventSource);

	friend class IODMAController;

public:
	typedef void (*Action)(OSObject *owner, IODMAEventSource *dmaES, IODMACommand *dmaCommand, IOReturn status, IOByteCount actualByteCount, AbsoluteTime timeStamp);
#define IODMAEventAction IODMAEventSource::Action

protected:
	virtual void completeDMACommand(IODMACommand *dmaCommand);
	virtual void notifyDMACommand(IODMACommand *dmaCommand, IOReturn status, IOByteCount actualByteCount, AbsoluteTime timeStamp);

public:
	static OSPtr<IODMAEventSource> dmaEventSource(OSObject *owner,
	    IOService *provider,
	    Action completion = NULL,
	    Action notification = NULL,
	    UInt32 dmaIndex = 0);

	virtual IOReturn startDMACommand(IODMACommand *dmaCommand, IODirection direction, IOByteCount byteCount = 0, IOByteCount byteOffset = 0);
	virtual IOReturn stopDMACommand(bool flush = false, uint64_t timeout = UINT64_MAX);

	virtual IOReturn queryDMACommand(IODMACommand **dmaCommand, IOByteCount *transferCount, bool waitForIdle = false);

	virtual IOByteCount getFIFODepth(IODirection direction = kIODirectionNone);
	virtual IOReturn setFIFODepth(IOByteCount depth);
	virtual IOByteCount validFIFODepth(IOByteCount depth, IODirection direction);

	virtual IOReturn setFrameSize(UInt8 byteCount);

	virtual IOReturn setDMAConfig(UInt32 dmaIndex);
	virtual bool validDMAConfig(UInt32 dmaIndex);

private:
	IOService       *dmaProvider;
	OSPtr<IODMAController> dmaController;
	UInt32          dmaIndex;
	queue_head_t    dmaCommandsCompleted;
	IOSimpleLock    *dmaCommandsCompletedLock;
	Action          dmaCompletionAction;
	Action          dmaNotificationAction;
	bool            dmaSynchBusy;

	virtual bool init(OSObject *owner,
	    IOService *provider,
	    Action completion = NULL,
	    Action notification = NULL,
	    UInt32 dmaIndex = 0);
	virtual bool checkForWork(void) APPLE_KEXT_OVERRIDE;
	virtual void free(void) APPLE_KEXT_OVERRIDE;
};

#endif /* _IOKIT_IODMAEVENTSOURCE_H */
