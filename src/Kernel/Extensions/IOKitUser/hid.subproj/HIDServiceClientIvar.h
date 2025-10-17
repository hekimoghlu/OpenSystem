/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 14, 2022.
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
//  HIDServiceClientIvar.h
//  iohidobjc
//
//  Created by dekom on 10/5/18.
//

#ifndef HIDServiceClientIvar_h
#define HIDServiceClientIvar_h

#import <IOKit/hidobjc/hidobjcbase.h>
#import <CoreFoundation/CoreFoundation.h>
#import <objc/objc.h> // for objc_object
#include <os/lock_private.h>

#define HIDServiceClientIvar \
IOHIDEventSystemClientRef   system; \
CFTypeRef                   serviceID; \
os_unfair_recursive_lock    callbackLock; \
struct { \
    IOHIDServiceClientCallback  callback; \
    IOHIDServiceClientBlock     block; \
    void                        *target; \
    void                        *refcon; \
} removal; \
struct { \
    IOHIDVirtualServiceClientCallbacksV2  *callbacks; \
    void                                  *target; \
    void                                  *refcon; \
} virtualService; \
os_unfair_recursive_lock        serviceLock; \
CFMutableDictionaryRef          cachedProperties; \
IOHIDServiceFastPathInterface   **fastPathInterface; \
IOCFPlugInInterface             **plugInInterface; \
void                            *removalHandler; \
uint32_t                        primaryUsagePage; \
uint32_t                        primaryUsage; \
IOHIDServiceClientUsagePair     *usagePairs; \
uint32_t                        usagePairsCount;

typedef struct  {
    HIDServiceClientIvar
} HIDServiceClientStruct;

#endif /* HIDServiceClientIvar_h */
