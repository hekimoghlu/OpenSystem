/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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
#ifndef _IOKIT_IOSERVICESTATENOTIFICATIONEVENTSOURCE_H
#define _IOKIT_IOSERVICESTATENOTIFICATIONEVENTSOURCE_H

#include <libkern/c++/OSPtr.h>
#include <IOKit/IOEventSource.h>

class IOService;


/*! @class IOServiceStateNotificationEventSource : public IOEventSource
 *   @abstract Event source for IOService state notification delivery to work-loop based drivers.
 *   @discussion For use with the IOService.iig IOService::StateNotification* APIs.
 */

class IOServiceStateNotificationEventSource : public IOEventSource
{
	OSDeclareDefaultStructors(IOServiceStateNotificationEventSource);

public:
	typedef void (^ActionBlock)();

protected:
	IOService                    * fStateNotification;
	IOStateNotificationListenerRef fListener;
	bool                           fEnable;
	bool                           fArmed;

/*! @struct ExpansionData
 *   @discussion This structure will be used to expand the capablilties of the IOWorkLoop in the future.
 */

/*! @var reserved
 *   Reserved for future use.  (Internal use only)  */
	APPLE_KEXT_WSHADOW_PUSH;
	ExpansionData *reserved;
	APPLE_KEXT_WSHADOW_POP;

/*! @function free
 *   @abstract Sub-class implementation of free method, disconnects from the interrupt source. */
	virtual void free() APPLE_KEXT_OVERRIDE;

/*! @function checkForWork
 *   @abstract Pure Virtual member function used by IOWorkLoop for issueing a client calls.
 *   @discussion This function called when the work-loop is ready to check for any work to do and then to call out the owner/action.
 *   @result Return true if this function needs to be called again before all its outstanding events have been processed. */
	virtual bool checkForWork() APPLE_KEXT_OVERRIDE;

/*! @function setWorkLoop
 *   @abstract Sub-class implementation of setWorkLoop method. */
	virtual void setWorkLoop(IOWorkLoop *inWorkLoop) APPLE_KEXT_OVERRIDE;

public:

	/*!
	 * @brief       Create an IOServiceStateNotificationEventSource for notification of IOService state events sent by the StateNotificationSet() api.
	 * @param       service The object hosting state, typically returned by IOService::CopySystemStateNotificationService().
	 * @param       items Array of state item names to be notified about.
	 * @param       action Handler block to be invoked on the event source workloop when the notification fires.
	 * @return      IOServiceStateNotificationEventSource with +1 retain, or NULL on failure.
	 */
	static OSPtr<IOServiceStateNotificationEventSource>
	serviceStateNotificationEventSource(IOService *service,
	    OSArray * items,
	    ActionBlock action);

/*! @function enable
 *   @abstract Enable event source.
 *   @discussion A subclass implementation is expected to respect the enabled
 *  state when checkForWork is called.  Calling this function will cause the
 *  work-loop to be signalled so that a checkForWork is performed. */
	virtual void enable() APPLE_KEXT_OVERRIDE;

/*! @function disable
 *   @abstract Disable event source.
 *   @discussion A subclass implementation is expected to respect the enabled
 *  state when checkForWork is called. */
	virtual void disable() APPLE_KEXT_OVERRIDE;

private:
	OSMetaClassDeclareReservedUnused(IOServiceStateNotificationEventSource, 0);
	OSMetaClassDeclareReservedUnused(IOServiceStateNotificationEventSource, 1);
	OSMetaClassDeclareReservedUnused(IOServiceStateNotificationEventSource, 2);
	OSMetaClassDeclareReservedUnused(IOServiceStateNotificationEventSource, 3);
	OSMetaClassDeclareReservedUnused(IOServiceStateNotificationEventSource, 4);
	OSMetaClassDeclareReservedUnused(IOServiceStateNotificationEventSource, 5);
	OSMetaClassDeclareReservedUnused(IOServiceStateNotificationEventSource, 6);
	OSMetaClassDeclareReservedUnused(IOServiceStateNotificationEventSource, 7);
};

#endif /* !_IOKIT_IOSERVICESTATENOTIFICATIONEVENTSOURCE_H */
