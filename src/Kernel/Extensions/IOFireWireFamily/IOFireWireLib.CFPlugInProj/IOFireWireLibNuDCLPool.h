/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 1, 2025.
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
#import "IOFireWireLibIUnknown.h"

#import <IOKit/firewire/IOFireWireFamilyCommon.h>
#import <IOKit/firewire/IOFireWireLibIsoch.h>
#import <CoreFoundation/CoreFoundation.h>

namespace IOFireWireLib {

	class Device ;
	class NuDCL ;
	class CoalesceTree ;

	class NuDCLPool: public IOFireWireIUnknown
	{
		protected:
		
			Device &			fDevice ;
			DCLNuDCLLeader		fLeader ;
			CFMutableArrayRef	fProgram ;
			UInt8				fCurrentTag ;
			UInt8				fCurrentSync ;
	
		protected:

			NuDCLPool( const IUnknownVTbl & vTable, Device& device, UInt32 numDCLs ) ;
			virtual ~NuDCLPool() ;
	
			void						Free( NuDCLRef* dcl ) ;
			DCLCommand*					GetProgram( ) ;
			CFArrayRef					GetDCLs() ;
		
			void						PrintDCLs( NuDCLRef dcl ) ;
			void						PrintDCL( NuDCLRef dcl ) ;

			// Allocating
			NuDCL *						AppendDCL( 
												CFMutableSetRef		saveBag,
												NuDCL * 			dcl ) ;
			void						SetCurrentTagAndSync (
												UInt8 				tag, 
												UInt8 				sync ) ;
			NuDCLSendPacketRef			AllocateSendPacket ( 
												CFMutableSetRef 	saveBag, 
												UInt32 				numBuffers, 
												IOVirtualRange * 	buffers ) ;	
			NuDCLSkipCycleRef			AllocateSkipCycle (
												CFMutableSetRef 	saveBag = NULL ) ;
			NuDCLReceivePacketRef		AllocateReceivePacket ( 
												CFMutableSetRef 	saveBag, 
												UInt8 				headerBytes, 
												UInt32 				numBuffers, 
												IOVirtualRange * 	buffers ) ;

		public :

			IOByteCount					Export ( 
												IOVirtualAddress * 	outRanges,
												IOVirtualRange		bufferRanges[],
												unsigned			bufferRangeCount ) const ;
			void						CoalesceBuffers( 
												CoalesceTree & 		toTree ) const ;
			Device &					GetDevice() const						{ return fDevice ; }

	} ;
	
#pragma mark -
	class NuDCLPoolCOM: public NuDCLPool
	{
		private:
		
			static const IOFireWireNuDCLPoolInterface sInterface ;
			
	
		public:

			NuDCLPoolCOM( Device& device, UInt32 numDCLs ) ;
			virtual ~NuDCLPoolCOM() ;
			static const IUnknownVTbl **	Alloc( Device& device, UInt32 capacity ) ;
			virtual HRESULT					QueryInterface( REFIID iid, LPVOID* ppv ) ;
	
			static DCLCommand*				S_GetProgram( IOFireWireLibNuDCLPoolRef self ) ;		
			static CFArrayRef				S_GetDCLs( IOFireWireLibNuDCLPoolRef self ) ;
			static void						S_PrintProgram( IOFireWireLibNuDCLPoolRef self ) ;
			static void						S_PrintDCL( NuDCLRef dcl ) ;
			
			// Allocating NuDCLs:
		
			static void						S_SetCurrentTagAndSync( IOFireWireLibNuDCLPoolRef self, UInt8 tag, UInt8 sync ) ;
		
			static NuDCLSendPacketRef		S_AllocateSendPacket( IOFireWireLibNuDCLPoolRef self, CFMutableSetRef saveBag, UInt32 numRanges, IOVirtualRange* ranges ) ;
			static NuDCLSendPacketRef		S_AllocateSendPacket_v( IOFireWireLibNuDCLPoolRef self, CFMutableSetRef saveBag, IOVirtualRange* firstRange, ... ) ;
			static NuDCLSkipCycleRef		S_AllocateSkipCycle( IOFireWireLibNuDCLPoolRef self ) ;
			static NuDCLReceivePacketRef	S_AllocateReceivePacket( IOFireWireLibNuDCLPoolRef self, CFMutableSetRef saveBag, UInt8 headerBytes, UInt32 numBuffers, IOVirtualRange* buffers ) ;
			static NuDCLReceivePacketRef	S_AllocateReceivePacket_v( IOFireWireLibNuDCLPoolRef self, CFMutableSetRef saveBag, UInt8 headerBytes, IOVirtualRange* firstRange, ... ) ;

			// NuDCL configuration
		
//			static IOReturn					S_SetDCLNextDCL ( NuDCLRef dcl, NuDCLRef nextDCL ) ;
			static NuDCLRef					S_FindDCLNextDCL ( IOFireWireLibNuDCLPoolRef self, NuDCLRef dcl ) ;
			static IOReturn					S_SetDCLBranch ( NuDCLRef dcl, NuDCLRef branchDCL ) ;
			static NuDCLRef					S_GetDCLBranch ( NuDCLRef dcl ) ;
			static IOReturn					S_SetDCLTimeStampPtr ( NuDCLRef dcl, UInt32* timeStampPtr ) ;
			static UInt32*					S_GetDCLTimeStampPtr ( NuDCLRef dcl ) ;
			static IOReturn					S_SetDCLStatusPtr( NuDCLRef dcl, UInt32* statusPtr ) ;
			static UInt32*					S_GetDCLStatusPtr( NuDCLRef dcl ) ;
			static IOReturn					S_AddDCLRanges ( NuDCLRef dcl, UInt32 numRanges, IOVirtualRange* ranges ) ;
			static IOReturn					S_SetDCLRanges ( NuDCLRef dcl, UInt32 numRanges, IOVirtualRange* ranges ) ;
			static IOReturn					S_SetDCLRanges_v ( NuDCLRef dcl, IOVirtualRange* firstRange, ... ) ;
			static UInt32					S_GetDCLRanges ( NuDCLRef dcl, UInt32 maxRanges, IOVirtualRange* outRanges ) ;
			static UInt32					S_CountDCLRanges ( NuDCLRef dcl ) ;
			static IOReturn					S_GetDCLSpan ( NuDCLRef dcl, IOVirtualRange* spanRange ) ;
			static IOByteCount				S_GetDCLSize( NuDCLRef dcl ) ;
			
			static IOReturn					S_SetDCLCallback ( NuDCLRef dcl, NuDCLCallback callback ) ;
			static NuDCLCallback			S_GetDCLCallback ( NuDCLRef dcl ) ;
			static IOReturn					S_SetDCLUserHeaderPtr ( NuDCLRef dcl, UInt32 * headerPtr, UInt32 * mask ) ;
			static UInt32 *					S_GetDCLUserHeaderPtr ( NuDCLRef dcl ) ;
			static UInt32 *					S_GetDCLUserHeaderMaskPtr ( NuDCLRef dcl ) ;

			static void						S_SetDCLRefcon( NuDCLRef dcl, void* refcon ) ;
			static void*					S_GetDCLRefcon( NuDCLRef dcl ) ;
			
			static IOReturn					S_AppendDCLUpdateList( NuDCLRef dcl, NuDCLRef updateDCL ) ;
			static IOReturn					S_SetDCLUpdateList( NuDCLRef dcl, CFSetRef dclList ) ;
			static CFSetRef					S_GetDCLUpdateList( NuDCLRef dcl ) ;
			static IOReturn					S_EmptyDCLUpdateList( NuDCLRef dcl ) ;
			
			static IOReturn					S_SetDCLWaitControl( NuDCLRef dcl, Boolean wait ) ;
			static void						S_SetDCLFlags( NuDCLRef dcl, UInt32 flags ) ;
			static UInt32					S_GetDCLFlags( NuDCLRef dcl ) ;
			static IOReturn					S_SetDCLSkipBranch( NuDCLRef dcl, NuDCLRef skipCycleDCL ) ;
			static NuDCLRef					S_GetDCLSkipBranch( NuDCLRef dcl ) ;
			static IOReturn					S_SetDCLSkipCallback( NuDCLRef dcl, NuDCLCallback callback ) ;
			static NuDCLCallback			S_GetDCLSkipCallback( NuDCLRef dcl ) ;
			static IOReturn					S_SetDCLSkipRefcon( NuDCLRef dcl, void * refcon ) ;
			static void *					S_GetDCLSkipRefcon( NuDCLRef dcl ) ;
			static IOReturn					S_SetDCLSyncBits( NuDCLRef dcl, UInt8 syncBits ) ;
			static UInt8					S_GetDCLSyncBits( NuDCLRef dcl ) ;
			static IOReturn					S_SetDCLTagBits( NuDCLRef dcl, UInt8 tagBits ) ;
			static UInt8					S_GetDCLTagBits( NuDCLRef dcl ) ;

	} ;

} // namespace
