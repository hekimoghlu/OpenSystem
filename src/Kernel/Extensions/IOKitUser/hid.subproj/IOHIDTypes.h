/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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

#ifndef _OPEN_SOURCE_

/*
 * Copyright (c) 1999-2008 Apple Computer, Inc.  All Rights Reserved.
 * 
 * @APPLE_LICENSE_HEADER_START@
 * 
 * This file contains Original Code and/or Modifications of Original Code
 * as defined in and that are subject to the Apple Public Source License
 * Version 2.0 (the 'License'). You may not use this file except in
 * compliance with the License. Please obtain a copy of the License at
 * http://www.opensource.apple.com/apsl/ and read it before using this
 * file.
 * 
 * The Original Code and all software distributed under the License are
 * distributed on an 'AS IS' basis, WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESS OR IMPLIED, AND APPLE HEREBY DISCLAIMS ALL SUCH WARRANTIES,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, QUIET ENJOYMENT OR NON-INFRINGEMENT.
 * Please see the License for the specific language governing rights and
 * limitations under the License.
 * 
 * @APPLE_LICENSE_HEADER_END@
 */

#ifndef _IOKIT_HID_HIDTYPES_H
#define _IOKIT_HID_HIDTYPES_H

#include <TargetConditionals.h>

__BEGIN_DECLS

CF_ASSUME_NONNULL_BEGIN
CF_IMPLICIT_BRIDGING_ENABLED

/*!
	@typedef IOHIDSessionRef
	This is the type of a reference to a IOHIDSession.  IOHIDSession is mutable by defualt.
*/
typedef struct CF_BRIDGED_TYPE(id) __IOHIDSession * IOHIDSessionRef;

/*!
	@typedef IOHIDServiceRef
	This is the type of a reference to a IOHIDService.  IOHIDService is mutable by defualt.
*/
typedef struct CF_BRIDGED_TYPE(id) __IOHIDService * IOHIDServiceRef;

/*!
	@typedef IOHIDNotificationRef
	This is the type of a reference to a IOHIDNotification.  IOHIDNotification is immutable by defualt.
*/
typedef const struct CF_BRIDGED_TYPE(id) __IOHIDNotification * IOHIDNotificationRef;

typedef void (*IOHIDMatchingServicesCallback)(void * _Nullable target, void * _Nullable refcon, void * _Nullable sender, CFArrayRef services);

typedef void (*IOHIDServiceCallback)(void * _Nullable target, void * _Nullable refcon, IOHIDServiceRef service);

CF_IMPLICIT_BRIDGING_DISABLED
CF_ASSUME_NONNULL_END

__END_DECLS

#endif /* _IOKIT_HID_HIDTYPES_H */

#endif /* _OPEN_SOURCE_ */