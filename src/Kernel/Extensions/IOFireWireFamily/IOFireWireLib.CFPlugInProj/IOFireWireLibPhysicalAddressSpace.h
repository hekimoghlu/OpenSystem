/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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
*  IOFireWireLibPhysicalAddressSpace.h
*  IOFireWireLib
*
*  Created by NWG on Fri Dec 08 2000.
*  Copyright (c) 2000 Apple Computer, Inc. All rights reserved.
*
*/

#import "IOFireWireLibIUnknown.h"
#import "IOFireWireLibPriv.h"

namespace IOFireWireLib {

	class Device ;
	class PhysicalAddressSpace: IOFireWireIUnknown
	{
		typedef ::IOFireWirePhysicalAddressSpaceInterface	Interface ;
	
		public:
			//
			// === COM =====================================
			//
		
			struct InterfaceMap
			{
				IUnknownVTbl *						pseudoVTable;
				PhysicalAddressSpace*	obj;
			};
		
			// interfaces
			static Interface	sInterface ;
		
			// QueryInterface
			virtual HRESULT				QueryInterface(
												REFIID 				iid, 
												LPVOID*				ppv) ;
			// static allocator
			static IUnknownVTbl**		Alloc(
											Device&	inUserClient,
											UserObjectHandle inKernPhysicalAddrSpaceRef,
											UInt32 					inSize, 
											void* 					inBackingStore, 
											UInt32 					inFlags) ;
		
			//
			// === STATIC METHODS ==========================						
			//
			static void					SGetPhysicalSegments(
												IOFireWireLibPhysicalAddressSpaceRef self,
												UInt32*				ioSegmentCount,
												IOByteCount			outSegments[],
												IOPhysicalAddress	outAddresses[]) ;
			static IOPhysicalAddress	SGetPhysicalSegment(
												IOFireWireLibPhysicalAddressSpaceRef self,
												IOByteCount 		offset,
												IOByteCount*		length) ;
			static IOPhysicalAddress	SGetPhysicalAddress(
												IOFireWireLibPhysicalAddressSpaceRef self) ;
		
			static void					SGetFWAddress(
												IOFireWireLibPhysicalAddressSpaceRef self,
												FWAddress*			outAddr ) ;
			static void*				SGetBuffer(
												IOFireWireLibPhysicalAddressSpaceRef self) ;
			static const UInt32			SGetBufferSize(
												IOFireWireLibPhysicalAddressSpaceRef self) ;
		
			// --- constructor/destructor ------------------
									PhysicalAddressSpace(
											Device& inUserClient,
											UserObjectHandle    inKernPhysicalAddrSpaceRef,
											UInt32 					inSize, 
											void* 					inBackingStore, 
											UInt32 					inFlags) ;
			virtual					~PhysicalAddressSpace() ;
			IOReturn				Init() ;
		
			// --- accessors -------------------------------
			virtual void				GetPhysicalSegments(
												UInt32*				ioSegmentCount,
												IOByteCount			outSegments[],
												IOPhysicalAddress	outAddresses[]) ;
			virtual IOPhysicalAddress	GetPhysicalSegment(
												IOByteCount 		offset,
												IOByteCount*		length) ;
		protected:
			// --- member data -----------------------------
			Device&	mUserClient ;
			UserObjectHandle		mKernPhysicalAddrSpaceRef ;
			UInt32							mSize ;
			void*							mBackingStore ;
			FWAddress						mFWAddress ;
			
			FWPhysicalSegment32 *				mSegments ;
			UInt32							mSegmentCount ;
	} ;
}
