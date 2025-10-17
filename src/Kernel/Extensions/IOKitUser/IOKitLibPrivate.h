/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 24, 2023.
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
#ifndef _IOKIT_IOKITLIBPRIVATE_H
#define _IOKIT_IOKITLIBPRIVATE_H

#include <sys/cdefs.h>
#include <CoreFoundation/CFMachPort.h>

__BEGIN_DECLS

#if !__has_feature(objc_arc)
struct IONotificationPort
{
    mach_port_t		masterPort;
    mach_port_t		wakePort;
    CFMachPortRef	cfmachPort;
    CFRunLoopSourceRef	source;
    dispatch_source_t	dispatchSource;
    int32_t			refcount;
};
typedef struct IONotificationPort IONotificationPort;
#endif

void
IODispatchCalloutFromCFMessage(
        CFMachPortRef port,
        void *msg,
        CFIndex size,
        void *info );

kern_return_t
iokit_user_client_trap(
                       io_connect_t	connect,
                       unsigned int	index,
                       uintptr_t p1,
                       uintptr_t p2,
                       uintptr_t p3,
                       uintptr_t p4,
                       uintptr_t p5,
                       uintptr_t p6 );

kern_return_t
IOServiceGetState(
	io_service_t    service,
	uint64_t *	state );

kern_return_t
IOServiceGetBusyStateAndTime(
	io_service_t    service,
	uint64_t *	state,
	uint32_t *	busy_state,
	uint64_t *	accumulated_busy_time);

// masks for getState()
enum {
    kIOServiceInactiveState	= 0x00000001,
    kIOServiceRegisteredState	= 0x00000002,
    kIOServiceMatchedState	= 0x00000004,
    kIOServiceFirstPublishState	= 0x00000008,
    kIOServiceFirstMatchState	= 0x00000010
};

kern_return_t
_IOServiceGetAuthorizationID(
	io_service_t    service,
	uint64_t *	authorizationID );

kern_return_t
_IOServiceSetAuthorizationID(
	io_service_t    service,
	uint64_t	authorizationID );

boolean_t
_IOObjectConformsTo(
	io_object_t	object,
	const io_name_t	className,
	uint64_t        options);

kern_return_t
_IOObjectGetClass(
	io_object_t	object,
	uint64_t        options,
	io_name_t       className);

CFStringRef
_IOObjectCopyClass(
	io_object_t     object,
	uint64_t        options);


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
 * IOKit.framework versions of the API's in IOService.iig
 *
 * see IOService.iig IOService::CopySystemStateNotificationService()
 */
kern_return_t
IOServiceCopySystemStateNotificationService(
        mach_port_t		masterPort,
        io_service_t  * service);

/*
 * see IOService.iig IOService::StateNotificationItemCreate()
 */
kern_return_t
IOServiceStateNotificationItemCreate(io_service_t service, CFStringRef itemName, CFDictionaryRef schema);

/*
 * see IOService.iig IOService::StateNotificationItemSet()
 */
kern_return_t
IOServiceStateNotificationItemSet(io_service_t service, CFStringRef itemName, CFDictionaryRef value);

/*
 * see IOService.iig IOService::StateNotificationItemCopy()
 */
kern_return_t
IOServiceStateNotificationItemCopy(io_service_t service, CFStringRef itemName, CFDictionaryRef * value);

__END_DECLS

#endif /* ! _IOKIT_IOKITLIBPRIVATE_H */
