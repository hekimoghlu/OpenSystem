/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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
 * Copyright (c) 1999 Apple Computer, Inc.  All rights reserved.
 *
 * HISTORY
 *
 */


#ifndef _IOKIT_APPLEMACIODEVICE_H
#define _IOKIT_APPLEMACIODEVICE_H

#include <IOKit/IOService.h>

class AppleMacIODevice : public IOService
{
	OSDeclareDefaultStructors(AppleMacIODevice);

private:
	struct ExpansionData { };
	ExpansionData *reserved;

public:
	virtual bool compareName( OSString * name, OSString ** matched = 0 ) const APPLE_KEXT_OVERRIDE;
	virtual IOService *matchLocation(IOService *client) APPLE_KEXT_OVERRIDE;
	virtual IOReturn getResources( void ) APPLE_KEXT_OVERRIDE;

	OSMetaClassDeclareReservedUnused(AppleMacIODevice, 0);
	OSMetaClassDeclareReservedUnused(AppleMacIODevice, 1);
	OSMetaClassDeclareReservedUnused(AppleMacIODevice, 2);
	OSMetaClassDeclareReservedUnused(AppleMacIODevice, 3);
};

#endif /* ! _IOKIT_APPLEMACIODEVICE_H */
