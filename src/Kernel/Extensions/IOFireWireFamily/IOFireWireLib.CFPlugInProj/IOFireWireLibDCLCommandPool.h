/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 25, 2025.
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
*  IOFireWireLibDCLPool.h
*  IOFireWireFamily
*
*  Created on Mon Mar 12 2001.
*  Copyright (c) 2001-2002 Apple Computer, Inc. All rights reserved.
*
*/

#import "IOFireWireLibIUnknown.h"
#import "IOFireWireLibPriv.h"
#import "IOFireWireLibIsoch.h"

#import <IOKit/IOKitLib.h>
#import <sys/types.h>

namespace IOFireWireLib {

	class Device ;
	class TraditionalDCLCommandPool : public IOFireWireIUnknown
	{
		protected:
		
			Device&				mUserClient ;
			CFMutableArrayRef	mFreeBlocks ;
			CFMutableArrayRef	mFreeBlockSizes ;
			CFMutableArrayRef	mAllocatedBlocks ;
			CFMutableArrayRef	mAllocatedBlockSizes ;	
			UInt8*				mStorage ;
			IOByteCount			mStorageSize ;
			IOByteCount			mBytesRemaining ;
			pthread_mutex_t		mMutex ;
		
		public:
		
			TraditionalDCLCommandPool( const IUnknownVTbl & interface, Device& inUserClient, IOByteCount inSize ) ;
			virtual ~TraditionalDCLCommandPool() ;
			
		public:
		
			DCLCommand*			Allocate( IOByteCount size ) ;
			IOReturn			AllocateWithOpcode( DCLCommand* dcl, DCLCommand** outDCL, UInt32 opcode, ... ) ;
			DCLCommand*			AllocateTransferPacketDCL( DCLCommand* dcl, UInt32 opcode, void* buffer, IOByteCount size ) ;
			DCLCommand*			AllocateTransferBufferDCL( DCLCommand* dcl, UInt32 opcode, void* buffer, IOByteCount size, IOByteCount packetSize, UInt32 bufferOffset ) ;
			DCLCommand*			AllocateSendPacketStartDCL(
												DCLCommand* 		inDCL, 
												void*					inBuffer,
												IOByteCount				inSize) ;
			DCLCommand*	AllocateSendPacketWithHeaderStartDCL(
												DCLCommand* 		inDCL, 
												void*					inBuffer,
												IOByteCount				inSize) ;
			DCLCommand*	AllocateSendBufferDCL(		// currently does nothing
												DCLCommand* 		inDCL, 
												void*					inBuffer,
												IOByteCount				inSize,
												IOByteCount				inPacketSize,
												UInt32					inBufferOffset) ;
			DCLCommand*	AllocateSendPacketDCL(
												DCLCommand* 		inDCL,
												void*					inBuffer,
												IOByteCount				inSize) ;
			DCLCommand*	AllocateReceivePacketStartDCL(
												DCLCommand* 		inDCL, 
												void*					inBuffer,
												IOByteCount				inSize) ;
			DCLCommand*	AllocateReceivePacketDCL(
												DCLCommand* 		inDCL,
												void*					inBuffer,
												IOByteCount				inSize) ;
			DCLCommand*	AllocateReceiveBufferDCL(	// currently does nothing
												DCLCommand* 		inDCL, 
												void*					inBuffer,
												IOByteCount				inSize,
												IOByteCount				inPacketSize,
												UInt32					inBufferOffset) ;
			DCLCommand*			AllocateCallProcDCL( DCLCommand* inDCL, DCLCallCommandProc* proc, DCLCallProcDataType procData) ;
			DCLCommand*			AllocateLabelDCL( DCLCommand* dcl ) ;
			DCLCommand*			AllocateJumpDCL( DCLCommand* dcl, DCLLabel* pInJumpDCLLabel ) ;
			DCLCommand*			AllocateSetTagSyncBitsDCL( DCLCommand* dcl, UInt16 tagBits, UInt16 syncBits ) ;
			DCLCommand*			AllocateUpdateDCLListDCL( DCLCommand* dcl, DCLCommand** dclCommandList, UInt32 numCommands ) ;
			DCLCommand*			AllocatePtrTimeStampDCL( DCLCommand* dcl, UInt32* timeStampPtr ) ;
			void 				Free( DCLCommand* dcl ) ;
			IOByteCount			GetSize() ;
			Boolean				SetSize( IOByteCount size ) ;
			IOByteCount			GetBytesRemaining() ;

		protected:
		
			void				Lock() ;
			void				Unlock() ;
			void				CoalesceFreeBlocks() ;
	} ;
	
	
	class TraditionalDCLCommandPoolCOM: public TraditionalDCLCommandPool
	{
		typedef IOFireWireLibDCLCommandPoolRef 		Ref ;
		typedef IOFireWireDCLCommandPoolInterface	Interface ;

		public:
			TraditionalDCLCommandPoolCOM( Device& inUserClient, IOByteCount inSize ) ;
			virtual ~TraditionalDCLCommandPoolCOM() ;
		
			//
			// --- COM ---------------
			//		
			static Interface			sInterface ;
		
			//
			// --- IUNKNOWN support ----------------
			//
			static IUnknownVTbl**		Alloc(
												Device&	inUserClient, 
												IOByteCount						inSize) ;
			virtual HRESULT				QueryInterface(REFIID iid, void ** ppv ) ;
		
			//
			// --- static methods ------------------
			//
			static DCLCommand*	SAllocate(
												IOFireWireLibDCLCommandPoolRef	self, 
												IOByteCount 					inSize ) ;
			static IOReturn				SAllocateWithOpcode(
												IOFireWireLibDCLCommandPoolRef 	self, 
												DCLCommand*				inDCL,
												DCLCommand**				outDCL,
												UInt32			 				opcode, ... ) ;
			static DCLCommand*	SAllocateTransferPacketDCL(
												IOFireWireLibDCLCommandPoolRef 	self, 
												DCLCommand*				inDCL,
												UInt32							inOpcode,
												void*							inBuffer,
												IOByteCount						inSize) ;
			static DCLCommand*	SAllocateTransferBufferDCL(
												IOFireWireLibDCLCommandPoolRef 	self, 
												DCLCommand* 				inDCL, 
												UInt32 							inOpcode, 
												void* 							inBuffer, 
												IOByteCount 					inSize, 
												IOByteCount 					inPacketSize, 
												UInt32 							inBufferOffset) ;
			static DCLCommand*	SAllocateSendPacketStartDCL(
												IOFireWireLibDCLCommandPoolRef 	self, 
												DCLCommand* 				inDCL, 
												void*							inBuffer,
												IOByteCount						inSize) ;
			static DCLCommand*	SAllocateSendPacketWithHeaderStartDCL(
												IOFireWireLibDCLCommandPoolRef 	self, 
												DCLCommand* 				inDCL, 
												void*							inBuffer,
												IOByteCount						inSize) ;
			static DCLCommand*	SAllocateSendBufferDCL(
												IOFireWireLibDCLCommandPoolRef 	self, 
												DCLCommand* 				inDCL, 
												void*							inBuffer,
												IOByteCount						inSize,
												IOByteCount						inPacketSize,
												UInt32							inBufferOffset) ;
			static DCLCommand*	SAllocateSendPacketDCL(
												IOFireWireLibDCLCommandPoolRef 	self, 
												DCLCommand* 				inDCL,
												void*							inBuffer,
												IOByteCount						inSize) ;
			static DCLCommand*	SAllocateReceivePacketStartDCL(
												IOFireWireLibDCLCommandPoolRef 	self, 
												DCLCommand* 				inDCL, 
												void*							inBuffer,
												IOByteCount						inSize) ;
			static DCLCommand*	SAllocateReceivePacketDCL(
												IOFireWireLibDCLCommandPoolRef 	self, 
												DCLCommand* 				inDCL,
												void*							inBuffer,
												IOByteCount						inSize) ;
			static DCLCommand*	SAllocateReceiveBufferDCL(
												IOFireWireLibDCLCommandPoolRef 	self, 
												DCLCommand* 				inDCL, 
												void*							inBuffer,
												IOByteCount						inSize,
												IOByteCount						inPacketSize,
												UInt32							inBufferOffset) ;
			static DCLCommand*		SAllocateCallProcDCL( IOFireWireLibDCLCommandPoolRef self, DCLCommand* dcl,  DCLCallCommandProc* proc, DCLCallProcDataType procData ) ;
			static DCLCommand*		SAllocateLabelDCL( IOFireWireLibDCLCommandPoolRef self, DCLCommand* dcl ) ;
			static DCLCommand*		SAllocateJumpDCL( IOFireWireLibDCLCommandPoolRef self, DCLCommand* dcl, DCLLabel* jumpDCLLabel) ;
			static DCLCommand*		SAllocateSetTagSyncBitsDCL(
												IOFireWireLibDCLCommandPoolRef 	self, 
												DCLCommand* 				inDCL, 
												UInt16							inTagBits,
												UInt16							inSyncBits) ;
			static DCLCommand*	SAllocateUpdateDCLListDCL(
												IOFireWireLibDCLCommandPoolRef 	self, 
												DCLCommand* 				inDCL, 
												DCLCommand**					inDCLCommandList,
												UInt32							inNumCommands) ;
			static DCLCommand*	SAllocatePtrTimeStampDCL(
												IOFireWireLibDCLCommandPoolRef 	self, 
												DCLCommand* 				inDCL, 
												UInt32*							inTimeStampPtr) ;
			static void 				SFree(
												IOFireWireLibDCLCommandPoolRef 	self, 
												DCLCommand* 				inDCL ) ;
			static IOByteCount			SGetSize(
												IOFireWireLibDCLCommandPoolRef 	self ) ;
			static Boolean				SSetSize(
												IOFireWireLibDCLCommandPoolRef 	self, 
												IOByteCount 					inSize ) ;
			static IOByteCount			SGetBytesRemaining(
												IOFireWireLibDCLCommandPoolRef 	self ) ;
	} ;
}
