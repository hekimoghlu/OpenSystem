/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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
//  IOHIDEventPrivate.h
//  IOKitUser
//
//  Created by dekom on 8/20/18.
//

#ifndef IOHIDEventPrivate_h
#define IOHIDEventPrivate_h

#include <IOKit/hid/IOHIDEvent.h>

CF_EXPORT
IOHIDEventRef _IOHIDEventCreate(CFAllocatorRef allocator,
                                CFIndex dataSize,
                                IOHIDEventType type,
                                uint64_t timeStamp,
                                IOOptionBits options);

CF_EXPORT
Boolean _IOHIDEventEqual(CFTypeRef cf1, CFTypeRef cf2);

CF_EXPORT
CFStringRef IOHIDEventCopyDescription(IOHIDEventRef event);

CF_EXPORT
CFStringRef _IOHIDEventDebugInfo(IOHIDEventRef event);

#endif /* IOHIDEventPrivate_h */
