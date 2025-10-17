/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 5, 2021.
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
 *  IOFireWireLibIRMAllocation.h
 *  IOFireWireFamily
 *
 *  Created by Andy on 02/06/07.
 *  Copyright (c) 2007 Apple Computer, Inc. All rights reserved.
 *
 *	$Log: not supported by cvs2svn $
 *	Revision 1.2  2007/02/15 22:02:38  ayanowit
 *	More fixes for new IRMAllocation stuff.
 *	
 *	Revision 1.1  2007/02/09 20:38:00  ayanowit
 *	New IRMAllocation files for user-space lib.
 *	
 *	
 */

#import "IOFireWireLibIUnknown.h"
#import "IOFireWireLibPriv.h"
#import "IOFireWireLib.h"

namespace IOFireWireLib {

	class Device ;
	class IRMAllocation: public IOFireWireIUnknown
	{
		public:
			IRMAllocation( const IUnknownVTbl& interface, 
						  Device& userclient, 
						  UserObjectHandle inKernIRMAllocationRef,
						  void* inCallBack = 0,
						  void* inRefCon = 0) ;
										
			virtual ~IRMAllocation() ;

		public:

			static void	LostProc( IOFireWireLibIRMAllocationRef refcon, IOReturn result, void** args, int numArgs) ;
		
			Boolean NotificationIsOn(IOFireWireLibIRMAllocationRef self ) ;
		
			Boolean TurnOnNotification(IOFireWireLibIRMAllocationRef self ) ;
		
			void TurnOffNotification(IOFireWireLibIRMAllocationRef self ) ;	
		
			void SetReleaseIRMResourcesOnFree (IOFireWireLibIRMAllocationRef self, Boolean doRelease ) ;
			IOReturn AllocateIsochResources(IOFireWireLibIRMAllocationRef self, UInt8 isochChannel, UInt32 bandwidthUnits);
			IOReturn DeallocateIsochResources(IOFireWireLibIRMAllocationRef self);
			Boolean AreIsochResourcesAllocated(IOFireWireLibIRMAllocationRef self, UInt8 *pAllocatedIsochChannel, UInt32 *pAllocatedBandwidthUnits);
		
			void SetRefCon(IOFireWireLibIRMAllocationRef self, void* refCon) ;
			void* GetRefCon(IOFireWireLibIRMAllocationRef self) ;
		
		protected:
			Boolean mNotifyIsOn ;
			Device& mUserClient ;
			UserObjectHandle mKernIRMAllocationRef ;
			IOFireWireLibIRMAllocationLostNotificationProc mLostHandler ;
			void* mUserRefCon ;
			IOFireWireLibIRMAllocationRef mRefInterface ;
	} ;
	
	class IRMAllocationCOM: public IRMAllocation
	{
			typedef ::IOFireWireLibIRMAllocationInterface	Interface ;
	
		public:
			IRMAllocationCOM( Device&					userclient,
									UserObjectHandle	inKernIRMAllocationRef,
									void*				inCallBack,
									void*				inRefCon ) ;
			
			virtual ~IRMAllocationCOM() ;
		
		private:
			static Interface sInterface ;

		public:
			static IUnknownVTbl**	Alloc(	Device&				inUserClient, 
											UserObjectHandle	inKernIRMAllocationRef,
											void*				inCallBack,
											void*				inRefCon );
											
			virtual HRESULT			QueryInterface( REFIID iid, void ** ppv ) ;
			
		protected:
		
			static Boolean SNotificationIsOn (IOFireWireLibIRMAllocationRef self ) ;
		
			static Boolean STurnOnNotification (IOFireWireLibIRMAllocationRef self ) ;
		
			static void STurnOffNotification (IOFireWireLibIRMAllocationRef self ) ;	
		
			static const void SSetReleaseIRMResourcesOnFree (IOFireWireLibIRMAllocationRef self, Boolean doRelease ) ;
		
			static IOReturn SAllocateIsochResources(IOFireWireLibIRMAllocationRef self, UInt8 isochChannel, UInt32 bandwidthUnits);
			static IOReturn SDeallocateIsochResources(IOFireWireLibIRMAllocationRef self);
			static Boolean SAreIsochResourcesAllocated(IOFireWireLibIRMAllocationRef self, UInt8 *pAllocatedIsochChannel, UInt32 *pAllocatedBandwidthUnits);
		
			static void SSetRefCon(IOFireWireLibIRMAllocationRef self, void* refCon) ;
			static void* SGetRefCon(IOFireWireLibIRMAllocationRef self) ;

	};
}