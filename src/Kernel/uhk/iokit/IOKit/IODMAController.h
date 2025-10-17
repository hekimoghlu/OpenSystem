/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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
#ifndef _IOKIT_IODMACONTROLLER_H
#define _IOKIT_IODMACONTROLLER_H

#include <IOKit/IODMACommand.h>
#include <IOKit/IODMAEventSource.h>
#include <IOKit/IOService.h>
#include <libkern/c++/OSPtr.h>

class IODMAEventSource;

class IODMAController : public IOService
{
	OSDeclareAbstractStructors(IODMAController);

	friend class IODMAEventSource;

private:
	IOService       *_provider;
	OSPtr<const OSSymbol> _dmaControllerName;

protected:
	virtual void registerDMAController(IOOptionBits options = 0);
	virtual IOReturn initDMAChannel(IOService *provider, IODMAEventSource *dmaES, UInt32 *dmaIndex, UInt32 reqIndex) = 0;
	virtual IOReturn startDMACommand(UInt32 dmaIndex, IODMACommand *dmaCommand, IODirection direction,
	    IOByteCount byteCount = 0, IOByteCount byteOffset = 0) = 0;
	virtual IOReturn stopDMACommand(UInt32 dmaIndex, bool flush = false, uint64_t timeout = UINT64_MAX) = 0;
	virtual void completeDMACommand(IODMAEventSource *dmaES, IODMACommand *dmaCommand);
	virtual void notifyDMACommand(IODMAEventSource *dmaES, IODMACommand *dmaCommand, IOReturn status, IOByteCount actualByteCount, AbsoluteTime timeStamp);
	virtual IOReturn queryDMACommand(UInt32 dmaIndex, IODMACommand **dmaCommand, IOByteCount *transferCount, bool waitForIdle = false) = 0;
	virtual IOByteCount getFIFODepth(UInt32 dmaIndex, IODirection direction) = 0;
	virtual IOReturn setFIFODepth(UInt32 dmaIndex, IOByteCount depth) = 0;
	virtual IOByteCount validFIFODepth(UInt32 dmaIndex, IOByteCount depth, IODirection direction) = 0;
	virtual IOReturn setFrameSize(UInt32 dmaIndex, UInt8 byteCount) = 0;
	virtual IOReturn setDMAConfig(UInt32 dmaIndex, IOService *provider, UInt32 reqIndex) = 0;
	virtual bool validDMAConfig(UInt32 dmaIndex, IOService *provider, UInt32 reqIndex) = 0;

public:
	static OSPtr<const OSSymbol> createControllerName(UInt32 phandle);
	static IODMAController *getController(IOService *provider, UInt32 dmaIndex);

	virtual bool start(IOService *provider) APPLE_KEXT_OVERRIDE;
};


#endif /* _IOKIT_IODMACONTROLLER_H */
