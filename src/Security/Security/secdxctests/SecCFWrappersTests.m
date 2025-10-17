/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 18, 2024.
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
#import <XCTest/XCTest.h>

#include <utilities/SecCFWrappers.h>

@interface SecCFWrappersTests : XCTestCase

@end

@implementation SecCFWrappersTests

- (void)testCreateStringByTrimmingCharactersInSet {
    NSCharacterSet *asciiDigits = [NSCharacterSet characterSetWithRange:NSMakeRange('0', '9')];

    NSString *noDigits = CFBridgingRelease(CFStringCreateByTrimmingCharactersInSet((__bridge CFStringRef)@"keys", (__bridge CFCharacterSetRef)asciiDigits));
    XCTAssertEqualObjects(noDigits, @"keys", "Should return string without digits as-is");

    NSString *leadingDigits = CFBridgingRelease(CFStringCreateByTrimmingCharactersInSet((__bridge CFStringRef)@"12keys", (__bridge CFCharacterSetRef)asciiDigits));
    XCTAssertEqualObjects(leadingDigits, @"keys", "Should trim leading digits");

    NSString *trailingDigits = CFBridgingRelease(CFStringCreateByTrimmingCharactersInSet((__bridge CFStringRef)@"keys34", (__bridge CFCharacterSetRef)asciiDigits));
    XCTAssertEqualObjects(trailingDigits, @"keys", "Should trim trailing digits");

    NSString *leadingAndTrailingDigits = CFBridgingRelease(CFStringCreateByTrimmingCharactersInSet((__bridge CFStringRef)@"12keys34", (__bridge CFCharacterSetRef)asciiDigits));
    XCTAssertEqualObjects(leadingAndTrailingDigits, @"keys", "Should trim leading and trailing digits");

    NSString *allDigits = CFBridgingRelease(CFStringCreateByTrimmingCharactersInSet((__bridge CFStringRef)@"1234", (__bridge CFCharacterSetRef)asciiDigits));
    XCTAssertEqual(allDigits.length, 0, "Should return empty string for all digits");
}

@end
