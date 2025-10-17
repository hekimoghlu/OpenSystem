/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 25, 2024.
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
#ifndef _IOKIT_IOETHERNET_CONTROLLER_USER_H
#define _IOKIT_IOETHERNET_CONTROLLER_USER_H

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

__BEGIN_DECLS

typedef struct __IOEthernetController * IOEthernetControllerRef;

typedef void (*IOEthernetControllerCallback)(IOEthernetControllerRef controller, void * refcon);

extern CFTypeRef kIOEthernetHardwareAddress;
extern CFTypeRef kIOUserEthernetInterfaceRole;

/*!
 * @const kIOUserEthernetInterfaceMergeProperties
 * @abstract
 * The key for a dictionary of properties to merge into the property table
 * of the Ethernet interface.
 * @discussion
 * The properties supplied to <code>IOEthernetControllerCreate</code> may
 * contain a dictionary stored using this key. The contents of the dictionary
 * are merged to the property table of the IOEthernetInterface when it is
 * initialized, before the interface object is registered and attached as
 * a child of the Ethernet controller.
 */
extern CFTypeRef kIOUserEthernetInterfaceMergeProperties;

/*!
	@function   IOEthernetControllerGetTypeID
	@abstract   Returns the type identifier of all IOUserEthernet instances.
*/
CF_EXPORT
CFTypeID IOEthernetControllerGetTypeID(void);

CF_EXPORT
IOEthernetControllerRef IOEthernetControllerCreate(
                                CFAllocatorRef                  allocator, 
                                CFDictionaryRef                 properties);

CF_EXPORT
io_object_t IOEthernetControllerGetIONetworkInterfaceObject(
                                IOEthernetControllerRef         controller);

CF_EXPORT
IOReturn    IOEthernetControllerSetLinkStatus(
                                IOEthernetControllerRef         controller, 
                                Boolean                         state);

CF_EXPORT
IOReturn    IOEthernetControllerSetPowerSavings(
								IOEthernetControllerRef			controller,
								Boolean							state);

CF_EXPORT
CFIndex     IOEthernetControllerReadPacket(
                                IOEthernetControllerRef         controller, 
                                uint8_t *                       buffer,
                                CFIndex                         bufferLength);

CF_EXPORT
IOReturn    IOEthernetControllerWritePacket(
                                IOEthernetControllerRef         controller, 
                                const uint8_t *                 buffer,
                                CFIndex                         bufferLength);

CF_EXPORT
void        IOEthernetControllerScheduleWithRunLoop(
                                IOEthernetControllerRef         controller, 
                                CFRunLoopRef                    runLoop,
                                CFStringRef                     runLoopMode);

CF_EXPORT
void        IOEthernetControllerUnscheduleFromRunLoop(
                                IOEthernetControllerRef         controller, 
                                CFRunLoopRef                    runLoop,
                                CFStringRef                     runLoopMode);

CF_EXPORT
void        IOEthernetControllerSetDispatchQueue(
                                IOEthernetControllerRef         controller, 
                                dispatch_queue_t                queue);

CF_EXPORT
void        IOEthernetControllerRegisterEnableCallback(
                                IOEthernetControllerRef         controller, 
                                IOEthernetControllerCallback    callback, 
                                void *                          refcon);

CF_EXPORT
void        IOEthernetControllerRegisterDisableCallback(
                                IOEthernetControllerRef         controller, 
                                IOEthernetControllerCallback    callback, 
                                void *                          refcon);

CF_EXPORT
void        IOEthernetControllerRegisterPacketAvailableCallback(
                                IOEthernetControllerRef         controller, 
                                IOEthernetControllerCallback    callback, 
                                void *                          refcon);

CF_EXPORT
void        IOEthernetControllerRegisterBSDAttachCallback(
                                IOEthernetControllerRef         controller,
                                IOEthernetControllerCallback    callback, 
                                void *                          refcon);

CF_EXPORT
int         IOEthernetControllerGetBSDSocket(
                                IOEthernetControllerRef         controller);

__END_DECLS

#endif /* _IOKIT_IOETHERNET_CONTROLLER_USER_H */
