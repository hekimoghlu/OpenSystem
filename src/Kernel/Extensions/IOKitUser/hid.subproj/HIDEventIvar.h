/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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
//  HIDEventIvar.h
//  IOKitUser
//
//  Created by dekom on 9/11/18.
//

#ifndef HIDEventIvar_h
#define HIDEventIvar_h

#import <IOKit/hidobjc/hidobjcbase.h>
#import <CoreFoundation/CoreFoundation.h>
#import <objc/objc.h> // for objc_object

/*
 * This is where we define the ivars that will be used by both the CF
 * IOHIDEventRef, and the objc HIDEvent. The variables must be the same between
 * the two objects to ensure proper bridging.
 */

#define HIDEventIvar \
uint64_t                timeStamp; /* Clock ticks from mach_absolute_time */ \
uint64_t                senderID; \
uint64_t                typeMask; \
uint32_t                options; \
uint8_t                 *attributeData; \
void                    *context; \
CFMutableDictionaryRef  attachments; \
CFTypeRef               sender; \
CFMutableArrayRef       children; \
IOHIDEventRef           parent; \
CFIndex                 attributeDataLength; \
CFIndex                 eventCount; \
IOHIDEventData          *eventData;

typedef struct  {
    HIDEventIvar
} HIDEventStruct;

#endif /* HIDEventIvar_h */
