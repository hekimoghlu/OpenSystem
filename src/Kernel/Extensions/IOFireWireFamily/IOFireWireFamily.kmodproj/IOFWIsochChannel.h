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
 * Copyright (c) 1999-2002 Apple Computer, Inc.  All rights reserved.
 *
 * HISTORY
 *
 */


#ifndef _IOKIT_IOFWISOCHCHANNEL_H
#define _IOKIT_IOFWISOCHCHANNEL_H

#include <libkern/c++/OSObject.h>
#include <IOKit/firewire/IOFireWireFamilyCommon.h>

enum
{
    kFWIsochChannelUnknownCondition	= 0,
    kFWIsochChannelNotEnoughBandwidth	= 1,
    kFWIsochChannelChannelNotAvailable	= 2
};

class IOFireWireController;
class IOFWIsochChannel;
class IOFWIsochPort;
class OSSet;
class IOFWReadQuadCommand;
class IOFWCompareAndSwapCommand;

/*! @class IOFWIsochChannel
*/
class IOFWIsochChannel : public OSObject
{
    OSDeclareDefaultStructors(IOFWIsochChannel)

	public:

		typedef IOReturn (ForceStopNotificationProc)(void* refCon, IOFWIsochChannel* channel, UInt32 stopCondition );

protected:
    IOFireWireController *			fControl;
    ForceStopNotificationProc* 		fStopProc;
    void *							fStopRefCon;
    IOFWIsochPort *					fTalker;
    OSSet *							fListeners;
    bool							fDoIRM;
    UInt32							fBandwidth;	// Allocation units used
    UInt32							fPacketSize;
    IOFWSpeed						fPrefSpeed;
    IOFWSpeed						fSpeed;		// Actual speed used
    UInt32							fChannel;	// Actual channel used
    IOFWReadQuadCommand *			fReadCmd;
    IOFWCompareAndSwapCommand *		fLockCmd;
    UInt32							fGeneration;	// When bandwidth was allocated
    
	IOLock *		fLock;
	
/*! @struct ExpansionData
    @discussion This structure will be used to expand the capablilties of the class in the future.
    */    
    struct ExpansionData { };

/*! @var reserved
    Reserved for future use.  (Internal use only)  */
    ExpansionData *reserved;

    static void					threadFunc( void * arg );
    
    virtual IOReturn			updateBandwidth(bool claim);
    virtual void				reallocBandwidth( UInt32 generation );	
    virtual void				free() APPLE_KEXT_OVERRIDE;

public:
    // Called from IOFireWireController
    virtual bool 				init( IOFireWireController *control, bool doIRM, UInt32 packetSize, 
										IOFWSpeed prefSpeed, ForceStopNotificationProc* stopProc,
										void *stopRefCon );
    virtual void 				handleBusReset();

    // Called by clients
    virtual IOReturn 			setTalker(IOFWIsochPort *talker);
    virtual IOReturn 			addListener(IOFWIsochPort *listener);

    virtual IOReturn 			allocateChannel();
    virtual IOReturn 			releaseChannel();
    virtual IOReturn 			start();
    virtual IOReturn 			stop();

protected:
	// handles IRM and channel determination and allocation.
	// called by both user and kernel isoch channels
	IOReturn					allocateChannelBegin( IOFWSpeed speed, UInt64 allowedChans, UInt32 * channel = NULL ) ;

	// handles IRM and channel allocation.
	// called by both user and kernel isoch channels
	IOReturn					releaseChannelComplete() ;

	IOReturn	checkMemoryInRange( IOMemoryDescriptor * memory );

private:
    OSMetaClassDeclareReservedUnused(IOFWIsochChannel, 0);
    OSMetaClassDeclareReservedUnused(IOFWIsochChannel, 1);
    OSMetaClassDeclareReservedUnused(IOFWIsochChannel, 2);
    OSMetaClassDeclareReservedUnused(IOFWIsochChannel, 3);

};

typedef IOFWIsochChannel::ForceStopNotificationProc 	FWIsochChannelForceStopNotificationProc ;
typedef IOFWIsochChannel::ForceStopNotificationProc* 	FWIsochChannelForceStopNotificationProcPtr ;

#endif /* ! _IOKIT_IOFWISOCHCHANNEL_H */

