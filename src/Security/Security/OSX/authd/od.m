/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 24, 2024.
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
//  od.m
//  authd
//
//  Copyright Â© 2021 Apple. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <OpenDirectory/OpenDirectory.h>
#import <opendirectory/odcustomfunctions.h>
#import <CoreAnalytics/CoreAnalytics.h>

#import "od.h"
#import "debugging.h"

AUTHD_DEFINE_LOG

bool odUserHasSecureToken(const char *username)
{
    NSError *localError = nil;
    NSString *recordName = [NSString stringWithUTF8String:username];
    if (!recordName) {
        os_log_error(AUTHD_LOG, "Unable to get username for %s", username);
        return false;
    }
    ODNode *node = [ODNode nodeWithSession:(__bridge ODSession *)(kODSessionDefault) type:kODNodeTypeLocalNodes error:&localError];
    if (node == nil) {
        os_log_error(AUTHD_LOG, "OD node failed with %{public}@", localError);
        return false;
    }
    
    NSDictionary *payload = @{ kODHasSecureTokenRecordnameKey : recordName };
    NSDictionary *result = (NSDictionary *)[node customFunction:kODCustomFunctionHasSecureToken payload:payload error:&localError];
    if ((!result) || localError) {
        os_log_error(AUTHD_LOG, "OD failed with %{public}@", localError);
        return false;
    }

    NSNumber *num = result[kODHasSecureTokenResultKey];
    return  num.boolValue == YES;
}

void analyticsSendEventLazy(CFStringRef eventName, CFDictionaryRef payload)
{
#if DEBUG
    os_log_debug(AUTHD_LOG, "analytics: { event: %@, payload: %@ }", eventName, payload);
#else
    AnalyticsSendEventLazy((__bridge const NSString * _Nonnull)(eventName), ^NSDictionary<NSString *,NSObject *> *{
        os_log_debug(AUTHD_LOG, "analytics: { event: %@, payload: %@ }", eventName, payload);
        return (__bridge NSDictionary<NSString *,NSObject *> *)(payload);
    });
#endif // DEBUG
}
