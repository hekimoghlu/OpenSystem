/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 13, 2023.
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
#import <mach/mach_time.h>

#import "OctagonSignPosts.h"

os_log_t _OctagonSignpostLogSystem(void) {
    static os_log_t log = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        log = os_log_create("com.apple.security.signposts", "signpost");
    });
    return log;
}

#pragma mark - Signpost Methods

OctagonSignpost _OctagonSignpostCreate(os_log_t subsystem) {
    os_signpost_id_t identifier = os_signpost_id_generate(subsystem);
    uint64_t timestamp = mach_continuous_time();
    return (OctagonSignpost){
        .identifier = identifier,
        .timestamp = timestamp,
    };
}

uint64_t _OctagonSignpostGetNanoseconds(OctagonSignpost signpost) {
    static struct mach_timebase_info timebase_info;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        mach_timebase_info(&timebase_info);
    });

    uint64_t interval = mach_continuous_time() - signpost.timestamp;

    return (uint64_t)(interval *
                      ((double)timebase_info.numer / timebase_info.denom));
}
