/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 9, 2024.
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
 *  IOFWUserIsochChannel.h
 *  IOFireWireFamily
 *
 *  Created by noggin on Tue May 15 2001.
 *  Copyright (c) 2001 Apple Computer, Inc. All rights reserved.
 *
 */

// public
#import <IOKit/firewire/IOFWIsochChannel.h>

class IOFWUserIsochChannel: public IOFWIsochChannel
{
	typedef IOFWIsochChannel super ;
	
	OSDeclareDefaultStructors(IOFWUserIsochChannel)
	
	public :
	
		virtual bool					init(	IOFireWireController *		control, 
													bool 						doIRM,
													UInt32 						packetSize, 
													IOFWSpeed 					prefSpeed ) ;

		// IOFWIsochChannel
		virtual IOReturn 				allocateChannel(void) APPLE_KEXT_OVERRIDE;
		virtual IOReturn 				releaseChannel(void) APPLE_KEXT_OVERRIDE;
		virtual IOReturn 				start(void) APPLE_KEXT_OVERRIDE;
		virtual IOReturn 				stop(void) APPLE_KEXT_OVERRIDE;
		
		// me
		IOReturn						allocateChannelBegin(
												IOFWSpeed		speed,
												UInt64			allowedChans,
												UInt32 *		outChannel )				{ return IOFWIsochChannel::allocateChannelBegin( speed, allowedChans, outChannel ) ; }
		IOReturn						releaseChannelComplete()							{ return IOFWIsochChannel::releaseChannelComplete() ; }
		IOReturn						allocateListenerPorts() ;
		IOReturn						allocateTalkerPort() ;
		static void						s_exporterCleanup( IOFWUserIsochChannel * channel ) ;
		
		inline io_user_reference_t *	getUserAsyncRef()									{ return fAsyncRef ; }
		inline void						setUserAsyncRef( OSAsyncReference64 asyncRef )		{ fAsyncRef = asyncRef ; }
		
	protected:
	
		bool					fBandwidthAllocated ;
		io_user_reference_t *   fAsyncRef ;

	public :

		static IOReturn						isochChannel_ForceStopHandler( void * self, IOFWIsochChannel*, UInt32 stopCondition ) ;
	
} ;
