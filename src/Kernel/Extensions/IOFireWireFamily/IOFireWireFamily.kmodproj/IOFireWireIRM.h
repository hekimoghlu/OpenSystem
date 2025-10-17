/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 26, 2025.
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
 * Copyright (c) 1999-2003 Apple Computer, Inc.  All rights reserved.
 *
 * IOFWIsochPort is an abstract object that represents hardware on the bus
 * (locally or remotely) that sends or receives isochronous packets.
 * Local ports are implemented by the local device driver,
 * Remote ports are implemented by the driver for the remote device.
 *
 * HISTORY
 * $Log: not supported by cvs2svn $
 * Revision 1.3  2003/07/21 06:52:59  niels
 * merge isoch to TOT
 *
 * Revision 1.1.14.2  2003/07/21 06:44:45  niels
 * *** empty log message ***
 *
 * Revision 1.1.14.1  2003/07/01 20:54:07  niels
 * isoch merge
 *
 *
 */


#ifndef _IOKIT_IOFIREWIREIRM_H
#define _IOKIT_IOFIREWIREIRM_H

#include <libkern/c++/OSObject.h>

#include <IOKit/firewire/IOFireWireController.h>
#include <IOKit/firewire/IOFireWireFamilyCommon.h>

class IOFireWireIRM : public OSObject
{
	OSDeclareAbstractStructors(IOFireWireIRM)

protected:
	
	IOFireWireController * 	fControl;
	
	UInt16			fIRMNodeID;
	UInt16			fOurNodeID;
	UInt32			fGeneration;
		
	// channel allocation	
	UInt32			fNewChannelsAvailable31_0;
	UInt32			fOldChannelsAvailable31_0;

	IOFWCompareAndSwapCommand * fLockCmd;
	bool						fLockCmdInUse;
	UInt32						fLockRetries;
	
	// broadcast channel register
	UInt32						fBroadcastChannelBuffer;
	IOFWPseudoAddressSpace *	fBroadcastChannelAddressSpace;	
	
public:

	static IOFireWireIRM * create( IOFireWireController * controller );

    virtual bool initWithController( IOFireWireController * control );
    virtual void free( void ) APPLE_KEXT_OVERRIDE;

	virtual bool isIRMActive( void );
	virtual void processBusReset( UInt16 ourNodeID, UInt16 irmNodeID, UInt32 generation );

protected:

	static void lockCompleteStatic( void *refcon, IOReturn status, IOFireWireNub *device, IOFWCommand *fwCmd );
	virtual void lockComplete( IOReturn status );
	virtual void allocateBroadcastChannel( void );
	
};

#endif
