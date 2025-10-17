/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 14, 2025.
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

#include <IOKit/IODMAController.h>
#include <libkern/c++/OSSharedPtr.h>


#define super IOService
OSDefineMetaClassAndAbstractStructors(IODMAController, IOService);

OSSharedPtr<const OSSymbol>
IODMAController::createControllerName(UInt32 phandle)
{
#define CREATE_BUF_LEN 48
	char           buf[CREATE_BUF_LEN];

	snprintf(buf, CREATE_BUF_LEN, "IODMAController%08X", (uint32_t)phandle);

	return OSSymbol::withCString(buf);
}

IODMAController *
IODMAController::getController(IOService *provider, UInt32 dmaIndex)
{
	OSData          *dmaParentData;
	OSSharedPtr<const OSSymbol> dmaParentName;
	IODMAController *dmaController;

	// Find the name of the parent dma controller
	OSSharedPtr<OSObject> prop = provider->copyProperty("dma-parent");
	dmaParentData = OSDynamicCast(OSData, prop.get());
	if (dmaParentData == NULL) {
		return NULL;
	}

	if (dmaParentData->getLength() == sizeof(UInt32)) {
		dmaParentName = createControllerName(*(UInt32 *)dmaParentData->getBytesNoCopy());
	} else {
		if (dmaIndex >= dmaParentData->getLength() / sizeof(UInt32)) {
			panic("dmaIndex out of range");
		}
		dmaParentName = createControllerName(*(UInt32 *)dmaParentData->getBytesNoCopy(dmaIndex * sizeof(UInt32), sizeof(UInt32)));
	}
	if (dmaParentName == NULL) {
		return NULL;
	}

	// Wait for the parent dma controller
	dmaController = OSDynamicCast(IODMAController, IOService::waitForService( IOService::nameMatching(dmaParentName.get()).detach()));

	return dmaController;
}


bool
IODMAController::start(IOService *provider)
{
	if (!super::start(provider)) {
		return false;
	}

	_provider = provider;

	return true;
}


// protected

void
IODMAController::registerDMAController(IOOptionBits options)
{
	OSData *phandleData;

	OSSharedPtr<OSObject> prop = _provider->copyProperty("AAPL,phandle");
	phandleData = OSDynamicCast(OSData, prop.get());

	_dmaControllerName = createControllerName(*(UInt32 *)phandleData->getBytesNoCopy());

	setName(_dmaControllerName.get());

	registerService(options | ((options & kIOServiceAsynchronous) ? 0 : kIOServiceSynchronous));
}

void
IODMAController::completeDMACommand(IODMAEventSource *dmaES, IODMACommand *dmaCommand)
{
	dmaES->completeDMACommand(dmaCommand);
}

void
IODMAController::notifyDMACommand(IODMAEventSource *dmaES, IODMACommand *dmaCommand, IOReturn status, IOByteCount actualByteCount, AbsoluteTime timeStamp)
{
	dmaES->notifyDMACommand(dmaCommand, status, actualByteCount, timeStamp);
}


// private
