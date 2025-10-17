/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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
#include <CoreFoundation/CoreFoundation.h>
#include <CoreFoundation/CFRuntime.h>
#include <xpc/xpc.h>
#import "config.h"

#if HEIMCRED_SERVER
#import <heim-ipc.h>
#import "heim_threads.h"

typedef enum {
    CRED_STATUS_ACQUIRE_INITIAL = -1,
    CRED_STATUS_ACQUIRE_START = 0,
    CRED_STATUS_ACQUIRE_STOPPED = 1,
    CRED_STATUS_ACQUIRE_FAILED = 2,
    CRED_STATUS_ACQUIRE_SUCCESS = 3
} cred_acquire_status;

#endif

#include "heimcred.h"

struct HeimMech;

#if HEIMCRED_SERVER
struct HeimCredEventContext_s {
    CFRuntimeBase runtime;
    HeimCredRef cred;
    HEIMDAL_MUTEX cred_mutex;  //protects the cred
};

typedef struct HeimCredEventContext_s *HeimCredEventContextRef;

CFTypeID HeimCredEventContextGetTypeID(void);

HeimCredEventContextRef HeimCredEventContextCreateItem(HeimCredRef cred);

#endif

struct HeimCred_s {
    CFRuntimeBase runtime;
    CFUUIDRef uuid;
    CFDictionaryRef attributes;
#if HEIMCRED_SERVER
    struct HeimMech *mech;
    
    HEIMDAL_MUTEX event_mutex;  //mutex for events, times, and statuses
    time_t renew_time;		//the next attempt to renew a renewable ticket
    heim_event_t renew_event;	//the event for renewal
    time_t next_acquire_time;  	//run time for next acquire attempt to get a new credential
    time_t expire;		//the time when the cred expires
    heim_event_t expire_event;	//the event for refreshing the cred
    cred_acquire_status acquire_status;	//the refresh status
    uid_t session;		//the session id for events;
    bool is_acquire_cred;	//used in expire event execution to either notify or acquire a new cred
    HeimCredEventContextRef renewEventContext;  //protected by event mutex
    HeimCredEventContextRef expireEventContext;  //protected by event mutex
#endif
};

typedef struct {
    dispatch_queue_t queue;
    CFTypeID haid;
#if HEIMCRED_SERVER
    CFTypeID heid;
    CFSetRef connections;
    CFMutableDictionaryRef sessions;
    CFMutableDictionaryRef challenges;
    CFMutableDictionaryRef mechanisms;
    CFMutableDictionaryRef schemas;
    CFMutableDictionaryRef globalSchema;
    pid_t session;
#else
    CFMutableDictionaryRef items;
    xpc_connection_t conn;
#endif
    bool needFlush;
    bool flushPending;
} HeimCredContext;

extern HeimCredContext HeimCredCTX;

void
_HeimCredInitCommon(void);

HeimCredRef
HeimCredCreateItem(CFUUIDRef uuid) CF_RETURNS_RETAINED;

CFTypeID
HeimCredGetTypeID(void);

CFUUIDRef
HeimCredCopyUUID(xpc_object_t object, const char *key);

CFTypeRef
HeimCredMessageCopyAttributes(xpc_object_t object, const char *key, CFTypeID type);

void
HeimCredMessageSetAttributes(xpc_object_t object, const char *key, CFTypeRef attrs);

void
HeimCredSetUUID(xpc_object_t object, const char *key, CFUUIDRef uuid);

#define HEIMCRED_CONST(_t,_c) extern const char * _c##xpc
#include "heimcred-const.h"
#undef HEIMCRED_CONST
