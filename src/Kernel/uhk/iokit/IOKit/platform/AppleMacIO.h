/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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
 * Copyright (c) 1998 Apple Computer, Inc.  All rights reserved.
 *
 * HISTORY
 *
 */


#ifndef _IOKIT_APPLEMACIO_H
#define _IOKIT_APPLEMACIO_H

#include <IOKit/IOService.h>

#include <IOKit/platform/AppleMacIODevice.h>

class AppleMacIO : public IOService
{
	OSDeclareAbstractStructors(AppleMacIO);

	IOService *         fNub;
	IOMemoryMap *       fMemory;

	struct ExpansionData { };
	ExpansionData *fReserved;

protected:
	virtual bool selfTest( void );

public:
	virtual bool start( IOService * provider ) APPLE_KEXT_OVERRIDE;

	virtual IOService * createNub( IORegistryEntry * from );

	virtual void processNub( IOService * nub );

	virtual void publishBelow( IORegistryEntry * root );

	virtual const char * deleteList( void );
	virtual const char * excludeList( void );

	virtual bool compareNubName( const IOService * nub, OSString * name,
	    OSString ** matched = 0 ) const;

	virtual IOReturn getNubResources( IOService * nub );

	OSMetaClassDeclareReservedUnused(AppleMacIO, 0);
	OSMetaClassDeclareReservedUnused(AppleMacIO, 1);
	OSMetaClassDeclareReservedUnused(AppleMacIO, 2);
	OSMetaClassDeclareReservedUnused(AppleMacIO, 3);
};

#endif /* ! _IOKIT_APPLEMACIO_H */
