/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 2, 2023.
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

	class PHYPacketListener : public IOFireWireIUnknown
	{		
		protected:
		
			static IOFireWireLibPHYPacketListenerInterface	sInterface;

			Device &						mUserClient;
			UserObjectHandle				mKernelRef;
			UInt32							mQueueCount;
			void *							mRefCon;
			
			IOFireWireLibPHYPacketCallback			mCallback;
			IOFireWireLibPHYPacketSkippedCallback	mSkippedCallback;
			UInt32									mFlags;
			Boolean									mNotifyIsOn;
			
		public:
			PHYPacketListener( Device& userClient, UInt32 queue_count );
			
			virtual ~PHYPacketListener( );

			static IUnknownVTbl**	Alloc(	Device& userclient, UInt32 queue_count );			
	
			virtual HRESULT				QueryInterface( REFIID iid, LPVOID* ppv );	
		
		protected:
			inline PHYPacketListener *	GetThis( IOFireWireLibPHYPacketListenerRef self )		
					{ return IOFireWireIUnknown::InterfaceMap<PHYPacketListener>::GetThis( self ); }
	
			static void SSetRefCon( IOFireWireLibPHYPacketListenerRef self, void * refCon );

			static void * SGetRefCon( IOFireWireLibPHYPacketListenerRef self );

			static void SSetListenerCallback(	IOFireWireLibPHYPacketListenerRef self, 
												IOFireWireLibPHYPacketCallback	callback );

			static void SSetSkippedPacketCallback(	IOFireWireLibPHYPacketListenerRef	self, 
													IOFireWireLibPHYPacketSkippedCallback	callback );

			static Boolean SNotificationIsOn( IOFireWireLibPHYPacketListenerRef self );

			static IOReturn STurnOnNotification( IOFireWireLibPHYPacketListenerRef self );
			IOReturn TurnOnNotification( IOFireWireLibPHYPacketListenerRef self );
	
			static void STurnOffNotification( IOFireWireLibPHYPacketListenerRef self );
			void TurnOffNotification( IOFireWireLibPHYPacketListenerRef self );
	
			static void SClientCommandIsComplete(	IOFireWireLibPHYPacketListenerRef		self,
													FWClientCommandID			commandID );

			static void SSetFlags( IOFireWireLibPHYPacketListenerRef	self,
								   UInt32								flags );
		
			static UInt32 SGetFlags( IOFireWireLibPHYPacketListenerRef self );
		
			static void SListenerCallback( IOFireWireLibPHYPacketListenerRef self, IOReturn result, void ** args, int numArgs );
			static void SSkippedCallback( IOFireWireLibPHYPacketListenerRef self, IOReturn result, void ** args, int numArgs );

		};

}