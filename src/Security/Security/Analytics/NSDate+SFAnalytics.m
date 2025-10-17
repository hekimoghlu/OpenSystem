/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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
#import "NSDate+SFAnalytics.h"

@implementation NSDate (SFAnalytics)

- (NSTimeInterval)bucketToRoundingFactor:(SFAnalyticsTimestampBucket)bucket
{
    switch (bucket) {
        case SFAnalyticsTimestampBucketSecond:
            return 1;
        case SFAnalyticsTimestampBucketMinute:
            return 60;
        case SFAnalyticsTimestampBucketHour:
            return 60 * 60;
    }
}

- (NSTimeInterval)timeIntervalSince1970WithBucket:(SFAnalyticsTimestampBucket)bucket
{
    NSTimeInterval mask = [self bucketToRoundingFactor:bucket];
    NSTimeInterval now = [[NSDate date] timeIntervalSince1970];
    return (NSTimeInterval)(now + (mask - ((long)now % (long)mask)));
}

@end

