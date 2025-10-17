/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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
#ifndef _IOWATCHDOGTIMER_H
#define _IOWATCHDOGTIMER_H

#include <IOKit/IOService.h>

class IOWatchDogTimer : public IOService
{
	OSDeclareAbstractStructors(IOWatchDogTimer);

protected:
	IONotifier *notifier;
	struct ExpansionData { };
	APPLE_KEXT_WSHADOW_PUSH;
	ExpansionData *reserved;
	APPLE_KEXT_WSHADOW_POP;

public:
	virtual bool start(IOService *provider) APPLE_KEXT_OVERRIDE;
	virtual void stop(IOService *provider) APPLE_KEXT_OVERRIDE;
	virtual IOReturn setProperties(OSObject *properties) APPLE_KEXT_OVERRIDE;
	virtual void setWatchDogTimer(UInt32 timeOut) = 0;

	OSMetaClassDeclareReservedUnused(IOWatchDogTimer, 0);
	OSMetaClassDeclareReservedUnused(IOWatchDogTimer, 1);
	OSMetaClassDeclareReservedUnused(IOWatchDogTimer, 2);
	OSMetaClassDeclareReservedUnused(IOWatchDogTimer, 3);
};

#endif /* !_IOWATCHDOGTIMER_H */
