/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 4, 2023.
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
 *  IOMIGMachPort.h
 *
 *  Created by Roberto Yepez on 1/29/09.
 *  Copyright 2009 Apple, Inc. All rights reserved.
 *
 */

#ifndef _IO_MIG_MACH_PORT_H_
#define _IO_MIG_MACH_PORT_H_

#include <dispatch/dispatch.h>
#include <CoreFoundation/CoreFoundation.h>
#include <mach/mach.h>

__BEGIN_DECLS

typedef struct __IOMIGMachPort * IOMIGMachPortRef;

typedef void (*IOMIGMachPortTerminationCallback)(IOMIGMachPortRef client, void * refcon);
typedef Boolean (*IOMIGMachPortDemuxCallback)(IOMIGMachPortRef client, mach_msg_header_t * request, mach_msg_header_t * reply, void *refcon);

    
CF_EXPORT
CFTypeID IOMIGMachPortGetTypeID(void);

CF_EXPORT
IOMIGMachPortRef IOMIGMachPortCreate(CFAllocatorRef allocator, CFIndex maxMessageSize, mach_port_t port);

CF_EXPORT
mach_port_t IOMIGMachPortGetPort(IOMIGMachPortRef migPort);

CF_EXPORT
void IOMIGMachPortRegisterTerminationCallback(IOMIGMachPortRef client, IOMIGMachPortTerminationCallback callback, void *refcon);

CF_EXPORT
void IOMIGMachPortRegisterDemuxCallback(IOMIGMachPortRef client, IOMIGMachPortDemuxCallback callback, void *refcon);

CF_EXPORT
void IOMIGMachPortScheduleWithRunLoop(IOMIGMachPortRef server, CFRunLoopRef runLoop, CFStringRef runLoopMode);

CF_EXPORT
void IOMIGMachPortUnscheduleFromRunLoop(IOMIGMachPortRef server, CFRunLoopRef runLoop, CFStringRef runLoopMode);

CF_EXPORT
void IOMIGMachPortScheduleWithDispatchQueue(IOMIGMachPortRef server, dispatch_queue_t queue);

CF_EXPORT
void IOMIGMachPortUnscheduleFromDispatchQueue(IOMIGMachPortRef server, dispatch_queue_t queue);

// PORT CACHE SUPPORT
CF_EXPORT
void IOMIGMachPortCacheAdd(mach_port_t port, CFTypeRef server);

CF_EXPORT
void IOMIGMachPortCacheRemove(mach_port_t port);

CF_EXPORT
CFTypeRef IOMIGMachPortCacheCopy(mach_port_t port);

__END_DECLS


#endif /* _IO_MIG_MACH_PORT_H_ */
