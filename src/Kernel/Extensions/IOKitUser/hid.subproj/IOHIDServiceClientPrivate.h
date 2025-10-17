/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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
//  IOHIDServiceClientPrivate.h
//  IOKitUser
//
//  Created by dekom on 8/31/18.
//

#ifndef IOHIDServiceClientPrivate_h
#define IOHIDServiceClientPrivate_h

#include <IOKit/hid/IOHIDServiceClient.h>

__BEGIN_DECLS

CF_ASSUME_NONNULL_BEGIN
CF_IMPLICIT_BRIDGING_ENABLED

CF_EXPORT
IOHIDServiceClientRef _Nullable _IOHIDServiceClientCreatePrivate(CFAllocatorRef _Nullable allocator);

CF_EXPORT
void _IOHIDServiceClientReleasePrivate(IOHIDServiceClientRef service);

CF_EXPORT
CFStringRef IOHIDServiceClientCopyDescription(IOHIDServiceClientRef service);

CF_IMPLICIT_BRIDGING_DISABLED
CF_ASSUME_NONNULL_END

__END_DECLS

#endif /* IOHIDServiceClientPrivate_h */
