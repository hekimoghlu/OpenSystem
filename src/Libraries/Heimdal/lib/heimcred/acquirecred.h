/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 17, 2025.
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
#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#import "common.h"
#import "heimbase.h"
#import "heimcred.h"
#import "gsscred.h"
#import <heim-ipc.h>

void
renew_func(heim_event_t event, void *ptr);

void
expire_func(heim_event_t event, void *ptr);

void
final_func(void *ptr);

void
cred_update_acquire_status(HeimCredRef cred, int status);

void
cred_update_renew_time(HeimCredRef cred, bool is_retry);

void
cred_update_expire_time_locked(HeimCredRef cred, time_t t);

void
suspend_event_work_queue(void);

void
resume_event_work_queue(void);

void
_test_wait_for_event_work_queue(void);
