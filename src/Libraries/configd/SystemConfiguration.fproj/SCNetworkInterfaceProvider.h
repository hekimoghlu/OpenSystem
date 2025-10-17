/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 30, 2025.
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
#ifndef _SCNETWORKINTERFACEPROVIDER_H
#define _SCNETWORKINTERFACEPROVIDER_H

/*
 * Modification History
 *
 * January 17, 2018		Dieter Siegmund (dieter@apple.com)
 * - initial revision
 */

/*
 * SCNetworkInterfaceProvider.h
 */


#include <os/availability.h>
#include <CoreFoundation/CoreFoundation.h>
#include <SystemConfiguration/SCNetworkConfiguration.h>

__BEGIN_DECLS

typedef CF_ENUM(uint32_t, SCNetworkInterfaceProviderEvent) {
	kSCNetworkInterfaceProviderEventActivationRequested = 1,
	kSCNetworkInterfaceProviderEventActivationNoLongerRequested = 2,
};

typedef struct CF_BRIDGED_TYPE(id) __SCNetworkInterfaceProvider *
SCNetworkInterfaceProviderRef;

/*!
	@typedef SCNetworkInterfaceProviderEventHandler
	@discussion Event handler callback to process SCNetworkInterfaceProvider
	events.
	@param event The event to handle.
	@param event_data The event data, always NULL currently.
 */
typedef void
(^SCNetworkInterfaceProviderEventHandler)(SCNetworkInterfaceProviderEvent event,
					  CFDictionaryRef event_data);

/*!
	@function SCNetworkInterfaceProviderCreate
	@discussion  Create an interface provider for a single network
	interface. The interface provider processes the events on the
	interface and takes actions based on the specific event.
	After calling this function, activate the event handler by calling
	SCNetworkInterfaceProviderSetEventHandler() followed by
	SCNetworkInterfaceProviderResume().
	Calling CFRelease() will free resources and deactivate the
	SCNetworkInterfaceProvider callback.
	@param interfaceType The kSCNetworkInterfaceType that the interface
	provider handles e.g. kSCNetworkInterfaceTypeCellular.
	@param interfaceName The name of the network interface, e.g. "pdp_ip0".
	@param options NULL for now.
	@result A non-NULL SCNetworkInterfaceProviderRef if the interface
	provider was successfully registered, NULL otherwise.
 */
SCNetworkInterfaceProviderRef
SCNetworkInterfaceProviderCreate(CFStringRef interfaceType,
				 CFStringRef interfaceName,
				 CFDictionaryRef options)
     API_AVAILABLE(macos(10.14), ios(12.0));

/*!
	@function SCNetworkInterfaceProviderSetEventHandler
	@discussion  Set the event handler to process events for the
	SCNetworkInterfaceProvider object.
	@param provider The SCNetworkInterfaceProvider to set the callback for.
	@param handler The event handler to process events. Invoking this
	function more than once or with a NULL handler is not valid.
 */
void
SCNetworkInterfaceProviderSetEventHandler(SCNetworkInterfaceProviderRef provider,
					  SCNetworkInterfaceProviderEventHandler handler)
     API_AVAILABLE(macos(10.14), ios(12.0));

/*!
	@function SCNetworkInterfaceProviderResume
	@discussion  Activate the interface provider so that its event handler
	will get called.
	@param provider The provider object to enable events on.
 */
void
SCNetworkInterfaceProviderResume(SCNetworkInterfaceProviderRef provider)
     API_AVAILABLE(macos(10.14), ios(12.0));

__END_DECLS
#endif /* _SCNETWORKINTERFACEPROVIDER_H */
