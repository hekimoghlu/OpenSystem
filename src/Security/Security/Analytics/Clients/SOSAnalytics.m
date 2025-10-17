/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 5, 2022.
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
#if __OBJC2__

#import "SOSAnalytics.h"
#include <utilities/SecFileLocations.h>
#include <sys/stat.h>

NSString* const CKDKVSPerformanceCountersSampler = @"CKDKVSPerformanceCounterSampler";

CKDKVSPerformanceCounter* const CKDKVSPerfCounterSynchronize = (CKDKVSPerformanceCounter*)@"CKDKVS-synchronize";
CKDKVSPerformanceCounter* const CKDKVSPerfCounterSynchronizeWithCompletionHandler = (CKDKVSPerformanceCounter*)@"CKDKVS-synchronizeWithCompletionHandler";
CKDKVSPerformanceCounter* const CKDKVSPerfCounterIncomingMessages = (CKDKVSPerformanceCounter*)@"CKDKVS-incomingMessages";
CKDKVSPerformanceCounter* const CKDKVSPerfCounterOutgoingMessages = (CKDKVSPerformanceCounter*)@"CKDKVS-outgoingMessages";
CKDKVSPerformanceCounter* const CKDKVSPerfCounterTotalWaitTimeSynchronize = (CKDKVSPerformanceCounter*)@"CKDKVS-totalWaittimeSynchronize";
CKDKVSPerformanceCounter* const CKDKVSPerfCounterLongestWaitTimeSynchronize = (CKDKVSPerformanceCounter*)@"CKDKVS-longestWaittimeSynchronize";
CKDKVSPerformanceCounter* const CKDKVSPerfCounterSynchronizeFailures = (CKDKVSPerformanceCounter*)@"CKDKVS-synchronizeFailures";

@implementation SOSAnalytics

+ (NSString*)databasePath
{
    // This block exists because we moved database locations in 11.3 for easier sandboxing, so we're cleaning up.
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        WithPathInKeychainDirectory(CFSTR("sos_analytics.db"), ^(const char *filename) {
            remove(filename);
        });
        WithPathInKeychainDirectory(CFSTR("sos_analytics.db-wal"), ^(const char *filename) {
            remove(filename);
        });
        WithPathInKeychainDirectory(CFSTR("sos_analytics.db-shm"), ^(const char *filename) {
            remove(filename);
        });
    });
#if TARGET_OS_OSX
    return [SOSAnalytics defaultProtectedAnalyticsDatabasePath:@"sos_analytics"];
#else
    return [SOSAnalytics defaultAnalyticsDatabasePath:@"sos_analytics"];
#endif
}

+ (instancetype)logger
{
    return [super logger];
}

@end

#endif
