/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 9, 2024.
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
#ifndef _IOKIT_IOFIREWIREPOWERMANAGER_H
#define _IOKIT_IOFIREWIREPOWERMANAGER_H

#include <IOKit/firewire/IOFireWireFamilyCommon.h>

#include <libkern/c++/OSObject.h>
#include <IOKit/IOReturn.h>

class IOFireWireController;

/*! @class IOFireWirePowerManager
*/

class IOFireWirePowerManager : public OSObject
{
    OSDeclareAbstractStructors(IOFireWirePowerManager);

protected:
    
/*! @struct ExpansionData
    @discussion This structure will be used to expand the capablilties of the class in the future.
    */    
    struct ExpansionData { };

/*! @var reserved
    Reserved for future use.  (Internal use only)  */
    ExpansionData *reserved;

	IOFireWireController *	fControl;
	
	UInt32		fMaximumDeciwatts;
	UInt32		fAllocatedDeciwatts;

public:	
	static IOFireWirePowerManager * createWithController( IOFireWireController * controller );
	
	virtual bool initWithController( IOFireWireController * controller );

	virtual void setMaximumDeciwatts( UInt32 deciwatts );
	virtual IOReturn allocateDeciwatts( UInt32 deciwatts );
	virtual void deallocateDeciwatts( UInt32 deciwatts );
	
private:
    OSMetaClassDeclareReservedUnused(IOFireWirePowerManager, 0);
    OSMetaClassDeclareReservedUnused(IOFireWirePowerManager, 1);
    OSMetaClassDeclareReservedUnused(IOFireWirePowerManager, 2);
    OSMetaClassDeclareReservedUnused(IOFireWirePowerManager, 3);
    OSMetaClassDeclareReservedUnused(IOFireWirePowerManager, 4);
    OSMetaClassDeclareReservedUnused(IOFireWirePowerManager, 5);
    OSMetaClassDeclareReservedUnused(IOFireWirePowerManager, 6);
    OSMetaClassDeclareReservedUnused(IOFireWirePowerManager, 7);
    OSMetaClassDeclareReservedUnused(IOFireWirePowerManager, 8);
    OSMetaClassDeclareReservedUnused(IOFireWirePowerManager, 9);
};

#endif