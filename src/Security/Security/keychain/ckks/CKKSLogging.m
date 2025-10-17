/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 31, 2025.
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
#import "keychain/ckks/CKKS.h"

os_log_t CKKSLogObject(NSString* scope, NSString* _Nullable zoneName)
{
    __block os_log_t ret = OS_LOG_DISABLED;

    static dispatch_queue_t logQueue = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        logQueue = dispatch_queue_create("ckks-logger", DISPATCH_QUEUE_SERIAL_WITH_AUTORELEASE_POOL);
    });

    static NSMutableDictionary* scopeMap = nil;

    dispatch_sync(logQueue, ^{
        if(scopeMap == nil) {
            scopeMap = [NSMutableDictionary dictionary];
        }

        NSString* key = zoneName ? [scope stringByAppendingFormat:@"-%@", zoneName] : scope;

        ret = scopeMap[key];

        if(!ret) {
            ret = os_log_create("com.apple.security.ckks", [key cStringUsingEncoding:NSUTF8StringEncoding]);
            scopeMap[key] = ret;
        }
    });

    return ret;
}
