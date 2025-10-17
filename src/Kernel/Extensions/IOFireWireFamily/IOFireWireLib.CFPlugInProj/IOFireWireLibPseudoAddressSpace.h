/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 9, 2023.
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
 *  IOFireWirePseudoAddressSpacePriv.h
 *  IOFireWireLib
 *
 *  Created  by NWG on Wed Dec 06 2000.
 *  Copyright (c) 2000 Apple, Inc. All rights reserved.
 *
 */

#import "IOFireWireLibIUnknown.h"
#import "IOFireWireLibPriv.h"

namespace IOFireWireLib {

	class Device ;
	class PseudoAddressSpace: public IOFireWireIUnknown
	{
			typedef ::IOFireWirePseudoAddressSpaceInterface 	Interface ;
			typedef ::IOFireWireLibPseudoAddressSpaceRef		AddressSpaceRef ;
			typedef ::IOFireWirePseudoAddressSpaceWriteHandler	WriteHandler ;
			typedef ::IOFireWirePseudoAddressSpaceReadHandler 	ReadHandler ;
			typedef ::IOFireWirePseudoAddressSpaceSkippedPacketHandler SkippedPacketHandler ;
			
			// interfaces
			static Interface sInterface ;
		
		public:
			// static allocator
			static IUnknownVTbl** 	Alloc( Device& userclient, UserObjectHandle inKernAddrSpaceRef, 
											void* inBuffer, UInt32 inBufferSize, void* inBackingStore, 
											void* inRefCon) ;
		
			// QueryInterface
			virtual HRESULT	QueryInterface(REFIID iid, void **ppv );
		
			//
			// === STATIC METHODS ==========================						
			//
		
			static IOReturn							SInit() ;
			
			// callback management
			static const WriteHandler			SSetWriteHandler( AddressSpaceRef interface, WriteHandler inWriter ) ;
			static const ReadHandler			SSetReadHandler( AddressSpaceRef interface, ReadHandler inReader) ;
			static const SkippedPacketHandler	SSetSkippedPacketHandler( AddressSpaceRef interface, SkippedPacketHandler inHandler ) ;
		
			static Boolean			SNotificationIsOn(
											AddressSpaceRef interface) ;
			static Boolean			STurnOnNotification(
											AddressSpaceRef interface) ;
			static void				STurnOffNotification(
											AddressSpaceRef interface) ;	
			static void				SClientCommandIsComplete(
											AddressSpaceRef interface,	
											FWClientCommandID				commandID,
											IOReturn						status) ;
		
			// accessors
			static void				SGetFWAddress(
											AddressSpaceRef	interface,
											FWAddress*						outAddr) ;
			static void*			SGetBuffer(
											AddressSpaceRef	interface) ;
			static const UInt32		SGetBufferSize(
											AddressSpaceRef	interface) ;
			static void*			SGetRefCon(
											AddressSpaceRef	interface) ;
		
			// --- constructor/destructor ----------
									PseudoAddressSpace(
											Device&	userclient,
											UserObjectHandle				inKernAddrSpaceRef,
											void*							inBuffer,
											UInt32							inBufferSize,
											void*							inBackingStore,
											void*							inRefCon = 0) ;
			virtual					~PseudoAddressSpace() ;
					
			// --- callback methods ----------------
			static void				Writer( AddressSpaceRef refcon, IOReturn result, void** args,
											int numArgs) ;
			static void				Reader( AddressSpaceRef refcon, IOReturn result, void** args,
											int numArgs) ;
			static void				SkippedPacket( AddressSpaceRef refCon, IOReturn result, FWClientCommandID commandID,
											UInt32 packetCount) ;

			// --- notification methods ----------
			virtual const WriteHandler			SetWriteHandler( WriteHandler inWriter ) ;
			virtual const ReadHandler	 		SetReadHandler( ReadHandler inReader ) ;
			virtual const SkippedPacketHandler	SetSkippedPacketHandler( SkippedPacketHandler inHandler ) ;
			virtual Boolean						NotificationIsOn() const									{ return mNotifyIsOn ; } 
			virtual Boolean						TurnOnNotification( void* callBackRefCon ) ;
			virtual void						TurnOffNotification() ;
			virtual void						ClientCommandIsComplete( FWClientCommandID commandID, IOReturn status) ;
		
			virtual const FWAddress& 			GetFWAddress() ;
			virtual void*						GetBuffer() ;
			virtual const UInt32				GetBufferSize() ;
			virtual void*						GetRefCon() ;
		
			const ReadHandler					GetReader()	const											{ return mReader ; }
			const WriteHandler					GetWriter() const 											{ return mWriter ; }
			const SkippedPacketHandler			GetSkippedPacketHandler() const								{ return mSkippedPacketHandler ; }
			
		protected:
			// callback mgmt.
			Boolean					mNotifyIsOn ;
			CFRunLoopRef			mNotifyRunLoop ;
			IONotificationPortRef	mNotifyPort ;
			io_object_t				mNotify;		
			WriteHandler			mWriter ;
			ReadHandler				mReader ;
			SkippedPacketHandler	mSkippedPacketHandler ;
			Device&					mUserClient ;
			FWAddress				mFWAddress ;
			UserObjectHandle		mKernAddrSpaceRef ;
			char*					mBuffer ;
			UInt32					mBufferSize ;
		
			void*							mBackingStore ;
			void*							mRefCon ;
			
			CFMutableDictionaryRef			mPendingLocks ;
	} ;	
}
