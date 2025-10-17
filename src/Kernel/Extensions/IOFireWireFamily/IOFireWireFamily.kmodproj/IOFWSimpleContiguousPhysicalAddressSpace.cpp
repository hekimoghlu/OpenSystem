/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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
// public
#import <IOKit/IOLib.h>

// private
#include "IOFWSimpleContiguousPhysicalAddressSpace.h"
#include "FWDebugging.h"

// fun with binary compatibility

OSDefineMetaClassAndStructors( IOFWSimpleContiguousPhysicalAddressSpace, IOFWSimplePhysicalAddressSpace )

OSMetaClassDefineReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 0);
OSMetaClassDefineReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 1);
OSMetaClassDefineReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 2);
OSMetaClassDefineReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 3);
OSMetaClassDefineReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 4);
OSMetaClassDefineReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 5);
OSMetaClassDefineReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 6);
OSMetaClassDefineReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 7);
OSMetaClassDefineReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 8);
OSMetaClassDefineReservedUnused(IOFWSimpleContiguousPhysicalAddressSpace, 9);

struct MemberVariables
{
	FWAddress	fFWPhysicalAddress;
};

#define _members ((MemberVariables*)fSimpleContigPhysSpaceMembers)

// init
//
//

bool IOFWSimpleContiguousPhysicalAddressSpace::init( IOFireWireBus * control, vm_size_t size, IODirection direction )
{
	DebugLog("IOFWSimpleContiguousPhysicalAddressSpace<%p>::init\n", this );
	
	fSimpleContigPhysSpaceMembers = NULL;
	
	bool success = IOFWSimplePhysicalAddressSpace::init( control, size, direction, true );
		
	if( success )
	{
		IOReturn status = cachePhysicalAddress();
		if( status != kIOReturnSuccess )
			success = false;
	}
	
	return success;
}

// free
//
//

void IOFWSimpleContiguousPhysicalAddressSpace::free( void )
{
	IOFWSimplePhysicalAddressSpace::free();
}

// createMemberVariables
//
//

bool IOFWSimpleContiguousPhysicalAddressSpace::createMemberVariables( void )
{
	bool success = true;
	
	success = IOFWSimplePhysicalAddressSpace::createMemberVariables();
	
	if( success && (fSimpleContigPhysSpaceMembers == NULL) )
	{
		// create member variables
		
		if( success )
		{
			fSimpleContigPhysSpaceMembers = IOMallocType( MemberVariables );
			if( fSimpleContigPhysSpaceMembers == NULL )
				success = false;
		}
		
		// zero member variables
		
		if( success )
		{
			// largely redundant
			_members->fFWPhysicalAddress.nodeID = 0x0000;
			_members->fFWPhysicalAddress.addressHi = 0x0000;
			_members->fFWPhysicalAddress.addressLo = 0x00000000;
		}
		
		// clean up on failure
		
		if( !success )
		{
			destroyMemberVariables();
		}
	}
	
	return success;
}

// destroyMemberVariables
//
//

void IOFWSimpleContiguousPhysicalAddressSpace::destroyMemberVariables( void )
{
	IOFWSimplePhysicalAddressSpace::destroyMemberVariables();
	
	if( fSimpleContigPhysSpaceMembers != NULL )
	{
		IOFreeType( fSimpleContigPhysSpaceMembers, MemberVariables );
	}
}

#pragma mark -

// cachePhysicalAddress
//
// 

IOReturn IOFWSimpleContiguousPhysicalAddressSpace::cachePhysicalAddress( void )
{
	IOReturn status = kIOReturnSuccess;

	UInt32 segment_count = 0;
	FWSegment	segments[ 2 ];
	if( status == kIOReturnSuccess )
	{
		UInt64 offset_64 = 0;					
		segment_count = 2;
		status = getSegments( &offset_64, segments, &segment_count );
	}
	
	// sanity checks for contiguous allocation
	if( status == kIOReturnSuccess )
	{
		if( segment_count > 2 || segment_count == 0 )
		{
			status = kIOReturnNoResources;
		}
	}		
	
	if( status == kIOReturnSuccess )
	{
		if(  segments[0].length < getLength() )
		{
			status = kIOReturnNoResources;
		}
	}
	
	if( status == kIOReturnSuccess )
	{
		_members->fFWPhysicalAddress = segments[0].address;
	}
	
//	IOLog( "IOFWSimpleContiguousPhysicalAddressSpace::cachePhysicalAddress - 0x%04x %08lx\n", 
//					_members->fFWPhysicalAddress.addressHi, _members->fFWPhysicalAddress.addressLo );
					
	return status;
}

// getPhysicalAddress
//
//

FWAddress IOFWSimpleContiguousPhysicalAddressSpace::getFWAddress( void )
{
	return _members->fFWPhysicalAddress;
}
