/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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
//  HIDConnectionIvar.h
//  IOKitUser
//
//  Created by dekom on 9/16/18.
//

#ifndef HIDConnectionIvar_h
#define HIDConnectionIvar_h

#import <IOKit/hidobjc/hidobjcbase.h>
#import <CoreFoundation/CoreFoundation.h>
#import <objc/objc.h> // for objc_object
#import <os/lock_private.h>
#import <xpc/xpc.h>

#define HIDConnectionIvar \
IOHIDEventSystemRef                             system; \
CFMutableDictionaryRef                          notifications; \
IOHIDEventQueueRef                              queue; \
IOMIGMachPortRef                                port; \
mach_port_t                                     reply_port; \
IOHIDEventSystemConnectionDemuxCallback         demuxCallback; \
void                                            *demuxRefcon; \
IOHIDEventSystemConnectionTerminationCallback   terminationCallback; \
void                                            *terminationRefcon; \
CFMutableSetRef                                 services; \
pid_t                                           pid; \
dispatch_queue_t                                dispatchQueue; \
mach_port_t                                     sendPossiblePort; \
dispatch_source_t                               sendPossibleSource; \
dispatch_source_t                               replySendPossibleSource; \
boolean_t                                       sendPossible; \
CFMutableSetRef                                 propertySet; \
CFStringRef                                     caller; \
CFStringRef                                     procName; \
CFStringRef                                     uuid; \
const char                                      *uuidStr; \
int                                             type; \
CFDictionaryRef                                 attributes; \
task_t                                          task_name_port; \
audit_token_t                                   audit_token; \
os_unfair_recursive_lock                        lock; \
IOHIDEventSystemConnectionEntitlements          *entitlements; \
xpc_object_t                                    connectionEntitlements; \
boolean_t                                       disableProtectedServices; \
int                                             filterPriority; \
uint32_t                                        state; \
os_unfair_recursive_lock                        notificationsLock; \
CFMutableDictionaryRef                          virtualServices; \
uint64_t                                        eventFilterMask; \
uint32_t                                        eventFilteredCount; \
uint32_t                                        eventFilterTimeoutCount; \
uint32_t                                        droppedEventCount; \
uint32_t                                        currentDroppedEventCount; \
uint64_t                                        droppedEventTypeMask; \
uint32_t                                        eventCount; \
uint64_t                                        eventMask; \
struct timeval                                  lastDroppedEventTime; \
struct timeval                                  firstDroppedEventTime; \
uint64_t                                        maxEventLatency; \
IOReturn                                        droppedEventStatus; \
uint64_t                                        propertyChangeNotificationHandlingTime; \
IOHIDSimpleQueueRef                             eventLog; \
uint32_t                                        *eventTypeCnt; \
uint32_t                                        activityState; \
uint32_t                                        activityStateChangeCount; \
uint64_t                                        idleNotificationTime; \
dispatch_source_t                               activityDispatchSource; \
IOHIDNotificationRef                            activityNotification; \
IOHIDSimpleQueueRef                             activityLog; \
IOHIDConnectionFilterRef                        filter; \
boolean_t                                       serverDied; \

typedef struct  {
    HIDConnectionIvar
} HIDConnectionStruct;

#endif /* HIDConnectionIvar_h */
