/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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
#ifndef _IOFWSIMPLECONTIGUOUSPHYSICALADDRESSSPACE_H_
#define _IOFWSIMPLECONTIGUOUSPHYSICALADDRESSSPACE_H_

#include <IOKit/firewire/IOFWSimplePhysicalAddressSpace.h>

/*! @class IOFWSimpleContiguousPhysicalAddressSpace
*/

class IOFWSimpleContiguousPhysicalAddressSpace : public IOFWSimplePhysicalAddressSpace
{
	OSDeclareDefaultStructors( IOFWSimpleContiguousPhysicalAddressSpace )

private:

	void * fSimpleContigPhysSpaceMembers;
	
	IOReturn cachePhysicalAddress( void );

protected:	
	virtual bool createMemberVariables( void ) APPLE_KEXT_OVERRIDE;
	virtual void destroyMemberVariables( void ) APPLE_KEXT_OVERRIDE;

public:

	virtual bool init( IOFireWireBus * control, vm_size_t size, IODirection direction );
	virtual void free( void ) APPLE_KEXT_OVERRIDE;

	FWAddress getFWAddress( void );

private:
    OSMetaClassDeclareReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 0);
    OSMetaClassDeclareReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 1);
    OSMetaClassDeclareReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 2);
    OSMetaClassDeclareReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 3);
    OSMetaClassDeclareReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 4);
    OSMetaClassDeclareReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 5);
    OSMetaClassDeclareReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 6);
    OSMetaClassDeclareReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 7);
    OSMetaClassDeclareReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 8);
    OSMetaClassDeclareReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 9);

};

#endif
