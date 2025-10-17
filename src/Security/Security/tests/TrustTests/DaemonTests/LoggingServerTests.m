/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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
//  LoggingServerTests.m
//  Security
//
//

#include <AssertMacros.h>
#import <XCTest/XCTest.h>

#import "trust/trustd/SecTrustLoggingServer.h"

#import "TrustDaemonTestCase.h"

@interface LoggingServerTests : TrustDaemonTestCase
@end

@implementation LoggingServerTests

- (void)testIntegerTruncation {
    XCTAssertEqualObjects(TATruncateToSignificantFigures(5, 1), @(5));
    XCTAssertEqualObjects(TATruncateToSignificantFigures(5, 2), @(5));
    XCTAssertEqualObjects(TATruncateToSignificantFigures(42, 1), @(40));
    XCTAssertEqualObjects(TATruncateToSignificantFigures(42, 2), @(42));
    XCTAssertEqualObjects(TATruncateToSignificantFigures(-335, 1), @(-300));
    XCTAssertEqualObjects(TATruncateToSignificantFigures(-335, 2), @(-330));
    XCTAssertEqualObjects(TATruncateToSignificantFigures(-335, 3), @(-335));
    XCTAssertEqualObjects(TATruncateToSignificantFigures(12345678901LL, 2), @(12000000000LL));
    XCTAssertEqualObjects(TATruncateToSignificantFigures(12345678901LL, 7), @(12345670000LL));
    XCTAssertEqualObjects(TATruncateToSignificantFigures(-12345678901LL, 3), @(-12300000000LL));
}

@end
