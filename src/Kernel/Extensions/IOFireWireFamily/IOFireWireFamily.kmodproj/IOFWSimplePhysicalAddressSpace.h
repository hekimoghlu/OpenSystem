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
#ifndef _IOFWSIMPLEPHYSICALADDRESSSPACE_H_
#define _IOFWSIMPLEPHYSICALADDRESSSPACE_H_

#include <libkern/c++/OSObject.h>
#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/IODMACommand.h>
#include <IOKit/firewire/IOFWPhysicalAddressSpace.h>

/*! @class IOFWSimplePhysicalAddressSpace
*/

class IOFWSimplePhysicalAddressSpace : public IOFWPhysicalAddressSpace
{
	OSDeclareDefaultStructors( IOFWSimplePhysicalAddressSpace )
	
private:

	void *	fSimplePhysSpaceMembers;
		
	IOReturn allocateMemory( void );	
	void deallocateMemory( void );

protected:	
	virtual bool createMemberVariables( void );
	virtual void destroyMemberVariables( void );

public:

	virtual bool init( IOFireWireBus * control, vm_size_t size, IODirection direction, bool contiguous = false );
	virtual void free( void ) APPLE_KEXT_OVERRIDE;

	IOVirtualAddress getVirtualAddress( void );

private:
    OSMetaClassDeclareReservedUnused(IOFWSimplePhysicalAddressSpace, 0);
    OSMetaClassDeclareReservedUnused(IOFWSimplePhysicalAddressSpace, 1);
    OSMetaClassDeclareReservedUnused(IOFWSimplePhysicalAddressSpace, 2);
    OSMetaClassDeclareReservedUnused(IOFWSimplePhysicalAddressSpace, 3);
    OSMetaClassDeclareReservedUnused(IOFWSimplePhysicalAddressSpace, 4);
    OSMetaClassDeclareReservedUnused(IOFWSimplePhysicalAddressSpace, 5);
    OSMetaClassDeclareReservedUnused(IOFWSimplePhysicalAddressSpace, 6);
    OSMetaClassDeclareReservedUnused(IOFWSimplePhysicalAddressSpace, 7);
    OSMetaClassDeclareReservedUnused(IOFWSimplePhysicalAddressSpace, 8);
    OSMetaClassDeclareReservedUnused(IOFWSimplePhysicalAddressSpace, 9);
	
};

#endif
