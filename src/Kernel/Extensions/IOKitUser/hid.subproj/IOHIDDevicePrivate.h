/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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

//
//  IOHIDDevicePrivate.h
//  IOKitUser
//
//  Created by dekom on 8/31/18.
//

#ifndef IOHIDDevicePrivate_h
#define IOHIDDevicePrivate_h

#include <IOKit/hid/IOHIDDevice.h>

__BEGIN_DECLS

CF_ASSUME_NONNULL_BEGIN
CF_IMPLICIT_BRIDGING_ENABLED

CF_EXPORT
IOHIDDeviceRef _Nullable _IOHIDDeviceCreatePrivate(CFAllocatorRef _Nullable allocator);

CF_EXPORT
CFStringRef IOHIDDeviceCopyDescription(IOHIDDeviceRef device);

CF_EXPORT
void _IOHIDDeviceReleasePrivate(IOHIDDeviceRef device);

uint64_t IOHIDDeviceGetRegistryEntryID(IOHIDDeviceRef device);

CF_IMPLICIT_BRIDGING_DISABLED
CF_ASSUME_NONNULL_END

__END_DECLS

#endif /* IOHIDDevicePrivate_h */
