/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 29, 2023.
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
//  HIDSessionIvar.h
//  iohidobjc
//
//  Created by dekom on 9/13/18.
//

#ifndef HIDSessionIvar_h
#define HIDSessionIvar_h

#import <IOKit/hidobjc/hidobjcbase.h>
#import <CoreFoundation/CoreFoundation.h>
#import <objc/objc.h> // for objc_object

#define HIDSessionIvar \
IOHIDEventSystemRef         client; \
IOHIDEventCallback          callback; \
void                        *refCon; \
__IOHIDSessionQueueContext  *queueContext; \
pthread_cond_t              stateCondition; \
boolean_t                   state; \
boolean_t                   stateBusy; \
dispatch_queue_t            eventDispatchQueueSession; \
dispatch_source_t           eventDispatchSource; \
CFMutableArrayRef           eventDipsatchPending; \
CFMutableDictionaryRef      properties; \
uint32_t                    logLevel; \
CFMutableSetRef             serviceSet; \
CFMutableArrayRef           simpleSessionFilters; \
CFMutableArrayRef           sessionFilters; \
CFMutableArrayRef           pendingSessionFilters; \
uint64_t                    activityLastTimestamp; \
struct timeval              activityLastTime; \
CFMutableSetRef             activityNotificationSet;

typedef struct  {
    HIDSessionIvar
} HIDSessionStruct;

#endif /* HIDSessionIvar_h */
