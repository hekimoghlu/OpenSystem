/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 18, 2023.
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
//  NSDate+SFAnalyticsTests.m
//  KeychainAnalyticsTests
//

#import <XCTest/XCTest.h>

#import "../Analytics/NSDate+SFAnalytics.h"

@interface NSDate_SFAnalyticsTests : XCTestCase

@end

@implementation NSDate_SFAnalyticsTests

- (void)testCurrentTimeSeconds
{
    NSTimeInterval expectedTime = [[NSDate date] timeIntervalSince1970];
    NSTimeInterval actualTimeWithWiggle = [[NSDate date] timeIntervalSince1970];
    XCTAssertEqualWithAccuracy(actualTimeWithWiggle, expectedTime, 1, @"Expected to get roughly the same amount of seconds");
}

- (void)testCurrentTimeSecondsWithRounding
{
    NSTimeInterval factor = 3; // 3 seconds

    // Round into the same bucket
    NSTimeInterval now = [[NSDate date] timeIntervalSince1970];
    NSTimeInterval expectedTime = now + factor;
    NSTimeInterval actualTimeWithWiggle = [[NSDate date] timeIntervalSince1970WithBucket:SFAnalyticsTimestampBucketSecond];
    XCTAssertEqualWithAccuracy(actualTimeWithWiggle, expectedTime, factor, @"Expected to get roughly the same rounded time within the rounding factor");

    // Round into the next bucket
    now = [[NSDate date] timeIntervalSince1970];
    expectedTime = now + factor;
    sleep(factor);
    actualTimeWithWiggle = [[NSDate date] timeIntervalSince1970WithBucket:SFAnalyticsTimestampBucketSecond];
    XCTAssertEqualWithAccuracy(actualTimeWithWiggle, expectedTime, factor + 1, @"Expected to get roughly the same rounded time within the rounding factor");
}

@end
