/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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
#ifndef _IOKIT_IOFWASYNCSTREAMLISTENER_H
#define _IOKIT_IOFWASYNCSTREAMLISTENER_H

#include <IOKit/firewire/IOFireWireLink.h>
#include <IOKit/firewire/IOFWCommand.h>
#include <IOKit/IOBufferMemoryDescriptor.h>
#include <IOKit/firewire/IOFWDCLProgram.h>

class IOFWAsyncStreamReceiver;
class IOFWAsyncStreamReceivePort;

/*! @class IOFWAsyncStreamListener
*/
class IOFWAsyncStreamListener : public OSObject
{
    OSDeclareDefaultStructors(IOFWAsyncStreamListener)

friend class IOFWAsyncStreamReceiver;
friend class IOFireWireController;

public:

/*!	@function initAll
	@abstract Creates an AsyncStream client for an Isoch channel.
	@param control	Points to IOFireWireController.
	@param channel	Isoch channel for listening.
	@param proc		Callback on packet reception.
	@param obj  Client's callback object.
	@result returns true on success, else false.	*/	
	bool initAll( IOFireWireController *control, UInt32 channel, FWAsyncStreamReceiveCallback proc, void *refcon );

/*!	@function setListenerHandler
	@abstract Set the callback that should be called to handle incoming async stream packets
	@param inReceiver The callback to set.
	@result Returns the callback that was previously set or nil for none.*/
	const FWAsyncStreamReceiveCallback setListenerHandler( FWAsyncStreamReceiveCallback inReceiver );

/*!	@function TurnOffNotification
	@abstract Turns off client callback notification.
	@result   none.	*/	
	inline void TurnOffNotification() { fNotify = false; };

/*!	@function TurnOnNotification
	@abstract Turns on client callback notification.
	@result   none.	*/	
	inline void TurnOnNotification() { fNotify = true; };

/*!	@function IsNotificationOn
	@abstract checks the notification state.
	@result   true if ON, else false	*/	
	inline bool IsNotificationOn() { return fNotify; };

/*!	@function setFlags
	@abstract set flags for the listener.
	@param flags indicate performance metrics.
	@result none.	*/	
	void setFlags( UInt32 flags );
	
/*!	@function getFlags
	@abstract get the flags of listener.
	@param none.
	@result flags.	*/	
	UInt32 getFlags();

/*!	@function getRefCon
	@abstract get the refcon specific to this listener.
	@param none.
	@result fRefCon refcon passed during initialization. */	
	inline void* getRefCon() { return fRefCon; };
	
/*!	@function getOverrunCounter
	@abstract get overrun counter from the DCL program.
	@param none.
	@result returns the counter value.	*/	
	UInt32 getOverrunCounter();
	
protected:

	FWAsyncStreamReceiveCallback	fClientProc; 
	void							*fRefCon;
	IOFWAsyncStreamReceiver			*fReceiver;
	bool							 fNotify;
	UInt32							 fFlags;
	IOFireWireController			*fControl;

/*! @struct ExpansionData
    @discussion This structure will be used to expand the capablilties of the class in the future.
    */    
    struct ExpansionData { };

/*! @var reserved
    Reserved for future use.  (Internal use only)  */
    ExpansionData *reserved;
    
    virtual void		free(void) APPLE_KEXT_OVERRIDE;

private:
/*!	function getReceiver
	abstract Returns the Async Stream receiver object which tracks multiple
	         IOFWAsyncStreamListeners for the same Isoc channel. */	
	inline IOFWAsyncStreamReceiver *getReceiver() { return fReceiver; };

/*!	function invokeClients
	abstract Invokes client's callback function with fRefCon.	*/	
	void invokeClients( UInt8 *buffer );
	
    OSMetaClassDeclareReservedUnused(IOFWAsyncStreamListener, 0);
    OSMetaClassDeclareReservedUnused(IOFWAsyncStreamListener, 1);
};
#endif // _IOKIT_IOFWASYNCSTREAMLISTENER_H

