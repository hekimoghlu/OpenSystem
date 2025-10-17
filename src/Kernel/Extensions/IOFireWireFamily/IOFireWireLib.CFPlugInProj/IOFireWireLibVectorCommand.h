/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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
#import <IOKit/firewire/IOFireWireFamilyCommon.h>

#import "IOFireWireLibIUnknown.h"
#import "IOFireWireLib.h"
#import "IOFireWireLibPriv.h"

namespace IOFireWireLib 
{
	class Device;

	class VectorCommand : public IOFireWireIUnknown
	{		
		protected:
		
			static IOFireWireLibVectorCommandInterface	sInterface;
			static CFArrayCallBacks						sArrayCallbacks;

			Device &						mUserClient;
			UserObjectHandle				mKernCommandRef;
			void*							mRefCon;
			IOFireWireLibCommandCallback	mCallback;
			CFMutableArrayRef				mCommandArray;
			UInt32							mFlags;
			UInt32							mInflightCount;
			IOReturn						mStatus;
			
			CommandSubmitParams *			mSubmitBuffer;
			vm_size_t						mSubmitBufferSize;

			CommandSubmitResult *			mResultBuffer;
			vm_size_t						mResultBufferSize;
						
		public:
			VectorCommand(	Device &						userClient,
							IOFireWireLibCommandCallback	callback, 
							void *							refCon );
			
			virtual ~VectorCommand();

			static IUnknownVTbl**	Alloc(	Device& 						userclient,
											IOFireWireLibCommandCallback	callback,
											void*							inRefCon );			
	
			virtual HRESULT				QueryInterface( REFIID iid, LPVOID* ppv );	
		
		protected:
			inline VectorCommand *		GetThis( IOFireWireLibVectorCommandRef self )		
					{ return IOFireWireIUnknown::InterfaceMap<VectorCommand>::GetThis( self ); }

			static IOReturn SSubmit( IOFireWireLibVectorCommandRef self );
			virtual IOReturn Submit();

			static IOReturn SSubmitWithRefconAndCallback( IOFireWireLibVectorCommandRef self, void* refCon, IOFireWireLibCommandCallback inCallback );

			static Boolean SIsExecuting( IOFireWireLibVectorCommandRef self );

			static void SSetCallback( IOFireWireLibVectorCommandRef self, IOFireWireLibCommandCallback inCallback );			

			static void SVectorCompletionHandler(	void*				refcon,
													IOReturn			result,
													void*				quads[],
													UInt32				numQuads );

			virtual void VectorCompletionHandler(	IOReturn			result,
													void*				quads[],
													UInt32				numQuads );
																										
			static void SSetRefCon( IOFireWireLibVectorCommandRef self, void* refCon );

			static void * SGetRefCon( IOFireWireLibVectorCommandRef self );

			static void SSetFlags( IOFireWireLibVectorCommandRef self, UInt32 inFlags );

			static UInt32 SGetFlags( IOFireWireLibVectorCommandRef self );

			static IOReturn SEnsureCapacity( IOFireWireLibVectorCommandRef self, UInt32 capacity );
			virtual IOReturn EnsureCapacity( UInt32 capacity );

			static void SAddCommand( IOFireWireLibVectorCommandRef self, IOFireWireLibCommandRef command );
			
			static void SRemoveCommand( IOFireWireLibVectorCommandRef self, IOFireWireLibCommandRef command );
			
			static void SInsertCommandAtIndex( IOFireWireLibVectorCommandRef self, IOFireWireLibCommandRef command, UInt32 index );

			static IOFireWireLibCommandRef SGetCommandAtIndex( IOFireWireLibVectorCommandRef self, UInt32 index );

			static UInt32 SGetIndexOfCommand( IOFireWireLibVectorCommandRef self, IOFireWireLibCommandRef command );

			static void SRemoveCommandAtIndex( IOFireWireLibVectorCommandRef self, UInt32 index );

			static void SRemoveAllCommands( IOFireWireLibVectorCommandRef self );

			static UInt32 SGetCommandCount( IOFireWireLibVectorCommandRef self );

			static const void * SRetainCallback( CFAllocatorRef allocator, const void * value );
			static void SReleaseCallback( CFAllocatorRef allocator, const void * value );

	};

}