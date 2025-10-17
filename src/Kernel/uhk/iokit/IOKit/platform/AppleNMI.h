/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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
 * Copyright (c) 1998-9 Apple Computer, Inc.  All rights reserved.
 *
 *  DRI: Josh de Cesare
 *
 */

#ifndef _IOKIT_APPLENMI_H
#define _IOKIT_APPLENMI_H

#include <IOKit/IOService.h>
#include <IOKit/IOInterrupts.h>

// NMI Interrupt Constants
enum{
	kExtInt9_NMIIntSource      = 0x800506E0,
	kNMIIntLevelMask           = 0x00004000,
	kNMIIntMask                = 0x00000080
};


class AppleNMI : public IOService
{
	OSDeclareDefaultStructors(AppleNMI);

private:
	bool enable_debugger;
	bool mask_NMI;

	struct ExpansionData { };
	ExpansionData * reserved; // Reserved for future use

public:
	IOService *rootDomain;
	virtual bool start(IOService *provider) APPLE_KEXT_OVERRIDE;
	virtual IOReturn initNMI(IOInterruptController *parentController, OSData *parentSource);
	virtual IOReturn handleInterrupt(void *refCon, IOService *nub, int source);

// Power handling methods:
	virtual IOReturn powerStateWillChangeTo(IOPMPowerFlags, unsigned long, IOService*) APPLE_KEXT_OVERRIDE;

	OSMetaClassDeclareReservedUnused(AppleNMI, 0);
	OSMetaClassDeclareReservedUnused(AppleNMI, 1);
	OSMetaClassDeclareReservedUnused(AppleNMI, 2);
	OSMetaClassDeclareReservedUnused(AppleNMI, 3);
};

#endif /* ! _IOKIT_APPLENMI_H */
