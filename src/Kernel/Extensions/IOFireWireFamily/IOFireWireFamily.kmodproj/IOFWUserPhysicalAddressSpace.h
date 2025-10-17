/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 20, 2024.
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
 *  IOFWUserClientPhysAddrSpace.h
 *  IOFireWireFamily
 *
 *  Created by NWG on Fri Dec 08 2000.
 *  Copyright (c) 2000 Apple Computer, Inc. All rights reserved.
 *
 */

#ifndef _IOKIT_IOFWUserClientPhysAddrSpace_H_
#define _IOKIT_IOFWUserClientPhysAddrSpace_H_

#import <IOKit/firewire/IOFWAddressSpace.h>
#import "IOFireWireLibPriv.h"

class IOFWUserPhysicalAddressSpace: public IOFWPhysicalAddressSpace
{
	OSDeclareDefaultStructors(IOFWUserPhysicalAddressSpace)

	protected:
	
		UInt32					fSegmentCount;
		IOMemoryDescriptor *	fDescriptor;

	public:
	
		virtual void		free(void) APPLE_KEXT_OVERRIDE;
		static void			exporterCleanup( const OSObject * self );

		virtual bool 		initWithDesc(
									IOFireWireBus *			bus,
									IOMemoryDescriptor*		mem) APPLE_KEXT_OVERRIDE;
	
		// getters
		IOReturn			getSegmentCount( UInt32* outSegmentCount ) ;
		IOReturn			getSegments(
									UInt32*					ioSegmentCount,
									IOFireWireLib::FWPhysicalSegment32 outSegments[] ) ;
} ;

#endif //_IOKIT_IOFWUserClientPhysAddrSpace_H_
