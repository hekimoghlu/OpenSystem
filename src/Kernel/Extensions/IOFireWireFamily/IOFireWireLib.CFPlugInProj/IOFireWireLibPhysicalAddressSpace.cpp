/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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
 *  IOFireWireLibPhysicalAddressSpace.cpp
 *  IOFireWireLib
 *
 *  Created by NWG on Tue Dec 12 2000.
 *  Copyright (c) 2000 Apple Computer, Inc. All rights reserved.
 *
 */

#import "IOFireWireLibPhysicalAddressSpace.h"
#import "IOFireWireLibDevice.h"
#import <exception>

#import <System/libkern/OSCrossEndian.h>

namespace IOFireWireLib {
	
	// COM
	
	PhysicalAddressSpace::Interface PhysicalAddressSpace::sInterface =
	{
		INTERFACEIMP_INTERFACE,
		1, 0,	//vers, rev
		& PhysicalAddressSpace::SGetPhysicalSegments,
		& PhysicalAddressSpace::SGetPhysicalSegment,
		& PhysicalAddressSpace::SGetPhysicalAddress,
		
		SGetFWAddress,
		SGetBuffer,
		SGetBufferSize
	} ;
	
	HRESULT
	PhysicalAddressSpace::QueryInterface(REFIID iid, void **ppv)
	{
		HRESULT		result = S_OK ;
		*ppv = nil ;
	
		CFUUIDRef	interfaceID	= CFUUIDCreateFromUUIDBytes(kCFAllocatorDefault, iid) ;
	
		if ( CFEqual(interfaceID, IUnknownUUID) ||  CFEqual(interfaceID, kIOFireWirePhysicalAddressSpaceInterfaceID) )
		{
			*ppv = & GetInterface() ;
			AddRef() ;
		}
		else
		{
			*ppv = nil ;
			result = E_NOINTERFACE ;
		}	
		
		CFRelease(interfaceID) ;
		return result ;
	}
	
	IUnknownVTbl**
	PhysicalAddressSpace::Alloc(
		Device& 	inUserClient,
		UserObjectHandle		inAddrSpaceRef,
		UInt32		 					inSize, 
		void* 							inBackingStore, 
		UInt32 							inFlags)
	{
		PhysicalAddressSpace* me = nil ;
		try {
			me = new PhysicalAddressSpace(inUserClient, inAddrSpaceRef, inSize, inBackingStore, inFlags) ;
		} catch (...) {
		}

		return ( nil == me ) ? nil : reinterpret_cast<IUnknownVTbl**>(& me->GetInterface()) ;
	}
	
#pragma mark -
	void
	PhysicalAddressSpace::SGetPhysicalSegments(
		IOFireWireLibPhysicalAddressSpaceRef self,
		UInt32*				ioSegmentCount,
		IOByteCount			outSegments[],
		IOPhysicalAddress	outAddresses[])
	{
		IOFireWireIUnknown::InterfaceMap<PhysicalAddressSpace>::GetThis(self)->GetPhysicalSegments(ioSegmentCount, outSegments, outAddresses) ;
	}
	
	IOPhysicalAddress
	PhysicalAddressSpace::SGetPhysicalSegment(
		IOFireWireLibPhysicalAddressSpaceRef self,
		IOByteCount 		offset,
		IOByteCount*		length)
	{
		return IOFireWireIUnknown::InterfaceMap<PhysicalAddressSpace>::GetThis(self)->GetPhysicalSegment(offset, length) ;
	}
	
	IOPhysicalAddress
	PhysicalAddressSpace::SGetPhysicalAddress(
		IOFireWireLibPhysicalAddressSpaceRef self)
	{
		return IOFireWireIUnknown::InterfaceMap<PhysicalAddressSpace>::GetThis(self)->mSegments[0].location ;
	}
	
	void
	PhysicalAddressSpace::SGetFWAddress(IOFireWireLibPhysicalAddressSpaceRef self, FWAddress* outAddr )
	{
		bcopy(& IOFireWireIUnknown::InterfaceMap<PhysicalAddressSpace>::GetThis(self)->mFWAddress, outAddr, sizeof(*outAddr)) ;
	}
	
	void*
	PhysicalAddressSpace::SGetBuffer(IOFireWireLibPhysicalAddressSpaceRef self)
	{
		return IOFireWireIUnknown::InterfaceMap<PhysicalAddressSpace>::GetThis(self)->mBackingStore ;
	}
	
	const UInt32
	PhysicalAddressSpace::SGetBufferSize(IOFireWireLibPhysicalAddressSpaceRef self)
	{
		return IOFireWireIUnknown::InterfaceMap<PhysicalAddressSpace>::GetThis(self)->mSize ;
	}
	
#pragma mark -
	PhysicalAddressSpace::PhysicalAddressSpace( Device& inUserClient, UserObjectHandle inKernPhysicalAddrSpaceRef,
				UInt32 inSize, void* inBackingStore, UInt32 inFlags)
	: IOFireWireIUnknown( reinterpret_cast<const IUnknownVTbl &>( sInterface ) ),
	  mUserClient(inUserClient),
	  mKernPhysicalAddrSpaceRef(inKernPhysicalAddrSpaceRef),
	  mSize(inSize),
	  mBackingStore(inBackingStore),
	  mSegments( NULL ),
	  mSegmentCount(0)
	{
		inUserClient.AddRef() ;

		if (!mKernPhysicalAddrSpaceRef)
			throw kIOReturnNoMemory ;
		
		uint32_t outputCnt = 1;
		uint64_t outputVal = 0;
		IOReturn error = IOConnectCallScalarMethod( mUserClient.GetUserClientConnection(), 
													mUserClient.MakeSelectorWithObject( kPhysicalAddrSpace_GetSegmentCount_d, mKernPhysicalAddrSpaceRef ), 
													NULL,0,
													&outputVal,&outputCnt);
		mSegmentCount = outputVal & 0xFFFFFFFF;
		
		if ( error || mSegmentCount == 0)
			throw error ;
			
		mSegments = new FWPhysicalSegment32[mSegmentCount] ;
		if (!mSegments)
		{
			throw kIOReturnNoMemory ;
		}
		
		outputCnt = 1;
		outputVal = 0;
		const uint64_t inputs[3] = {(const uint64_t)mKernPhysicalAddrSpaceRef, mSegmentCount, (const uint64_t)mSegments};
		error = IOConnectCallScalarMethod( mUserClient.GetUserClientConnection(), 
										  kPhysicalAddrSpace_GetSegments, 
										  inputs,3,
										  &outputVal,&outputCnt);
		mSegmentCount = outputVal & 0xFFFFFFFF;
		
		if (error)
		{
			throw error ;
		}

#ifndef __LP64__		
		ROSETTA_ONLY(	
			{
				UInt32 i;
				for( i = 0; i < mSegmentCount; i++ )
				{
					mSegments[i].location = OSSwapInt32( mSegments[i].location );
					mSegments[i].length = OSSwapInt32( mSegments[i].length );
				}
			}
		);
#endif

		mFWAddress = FWAddress(0, mSegments[0].location, 0) ;
	}
	
	PhysicalAddressSpace::~PhysicalAddressSpace()
	{
		// call user client to delete our addr space ref here (if not yet released)
		uint32_t outputCnt = 0;
		const uint64_t inputs[1]={(const uint64_t)mKernPhysicalAddrSpaceRef};

		IOConnectCallScalarMethod(mUserClient.GetUserClientConnection(), 
								  kReleaseUserObject,
								  inputs,1,
								  NULL,&outputCnt);
		
		delete[] mSegments ;
	
		mUserClient.Release() ;
	}
	
	void
	PhysicalAddressSpace::GetPhysicalSegments(
		UInt32*				ioSegmentCount,
		IOByteCount			outSegmentLengths[],
		IOPhysicalAddress	outSegments[])
	{
		if ( !outSegments || !outSegmentLengths )
		{
			*ioSegmentCount = mSegmentCount ;
			return ;
		}
		
		*ioSegmentCount = MIN( *ioSegmentCount, mSegmentCount ) ;
		
		for( unsigned index=0; index < *ioSegmentCount; ++index )
		{
			outSegments[ index ] = mSegments[ index ].location ;
			outSegmentLengths[ index ] = mSegments[ index ].length ;
		}
	}
	
	IOPhysicalAddress
	PhysicalAddressSpace::GetPhysicalSegment(
		IOByteCount 		offset,
		IOByteCount*		length)
	{
		IOPhysicalAddress result = 0 ;
	
		if (mSegmentCount > 0)
		{		
			IOByteCount traversed = mSegments[0].length ;
			UInt32		currentSegment = 0 ;
			
			while((traversed <= offset) && (currentSegment < mSegmentCount))
			{
				traversed += mSegments[ currentSegment ].length ;
				++currentSegment ;
			}	
			
			if ( currentSegment <= mSegmentCount )
			{
				*length = mSegments[ currentSegment ].length ;
				result =  mSegments[ currentSegment ].location ;
			}
		}
		
		return result ;
	}
	
}
