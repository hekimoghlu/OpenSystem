/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 11, 2024.
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
/*
 * Copyright (c) 1998-2000 Apple Computer, Inc.  All rights reserved.
 *
 * HISTORY
 *
 */


#ifndef _IOKIT_APPLEPLATFORM_H
#define _IOKIT_APPLEPLATFORM_H

#include <IOKit/IOPlatformExpert.h>

enum {
	kBootROMTypeOldWorld = 0,
	kBootROMTypeNewWorld
};

enum {
	kChipSetTypePowerSurge = 0,
	kChipSetTypePowerStar,
	kChipSetTypeGossamer,
	kChipSetTypePowerExpress,
	kChipSetTypeCore99,
	kChipSetTypeCore2001
};

enum {
	kMachineTypeUnknown = 0
};

extern const OSSymbol *gGetDefaultBusSpeedsKey;

class ApplePlatformExpert : public IODTPlatformExpert
{
	OSDeclareAbstractStructors(ApplePlatformExpert);

private:
	SInt32 _timeToGMT;

	struct ExpansionData { };
	ExpansionData *reserved;

public:
	virtual bool start( IOService * provider ) APPLE_KEXT_OVERRIDE;
	virtual bool configure( IOService * provider ) APPLE_KEXT_OVERRIDE;
	virtual const char * deleteList( void ) APPLE_KEXT_OVERRIDE;
	virtual const char * excludeList( void ) APPLE_KEXT_OVERRIDE;

	virtual void registerNVRAMController( IONVRAMController * nvram ) APPLE_KEXT_OVERRIDE;

	virtual long getGMTTimeOfDay(void) APPLE_KEXT_OVERRIDE;
	virtual void setGMTTimeOfDay(long secs) APPLE_KEXT_OVERRIDE;

	virtual bool getMachineName(char *name, int maxLength) APPLE_KEXT_OVERRIDE;

	OSMetaClassDeclareReservedUnused(ApplePlatformExpert, 0);
	OSMetaClassDeclareReservedUnused(ApplePlatformExpert, 1);
	OSMetaClassDeclareReservedUnused(ApplePlatformExpert, 2);
	OSMetaClassDeclareReservedUnused(ApplePlatformExpert, 3);
};


#endif /* ! _IOKIT_APPLEPLATFORM_H */
