/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 16, 2021.
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
//  KCDerTest.m
//  Security
//
//

#import <XCTest/XCTest.h>

#import <Foundation/Foundation.h>
#import <KeychainCircle/KCDer.h>

@interface KCDerTest : XCTestCase

@end

@implementation KCDerTest

- (void) roundTripData: (NSData*) data {
    NSError* error = nil;
    size_t size = kcder_sizeof_data(data, &error);

    XCTAssert(size != 0, @"Bad size: %@", data);

    if (size == 0)
        return;

    NSMutableData *buffer = [NSMutableData dataWithLength:size];
    error = nil;
    uint8_t* beginning = kcder_encode_data(data, &error, [buffer mutableBytes], [buffer mutableBytes] + size);

    XCTAssert(beginning != NULL, "Error encoding: %@", error);

    if (beginning == NULL)
        return;

    XCTAssertEqual(beginning, [buffer mutableBytes], @"Size != buffer use");

    NSData* recovered = nil;

    error = nil;
    const uint8_t* end = kcder_decode_data(&recovered, &error, [buffer mutableBytes], [buffer mutableBytes] + size);

    XCTAssert(end != NULL, "Error decoding: %@", error);

    if (end == NULL)
        return;

    XCTAssertEqual(end, [buffer mutableBytes] + size, @"readback didn't use all the buffer");

    XCTAssertEqualObjects(data, recovered, @"Didn't get equal object");

}

- (void)testData {
    [self roundTripData: [NSData data]];

    uint8_t bytes[] = { 1, 2, 3, 0xFF, 4, 0x0, 0xA };
    [self roundTripData: [NSData dataWithBytes:bytes length:sizeof(bytes)]];
}

- (void) roundTripString: (NSString*) string {
    NSError* error = nil;

    size_t size = kcder_sizeof_string(string, &error);

    XCTAssert(size != 0, @"Bad size: %@", string);

    if (size == 0)
        return;

    NSMutableData *buffer = [NSMutableData dataWithLength:size];
    error = nil;
    uint8_t* beginning = kcder_encode_string(string, &error, [buffer mutableBytes], [buffer mutableBytes] + size);

    XCTAssert(beginning != NULL, "Error encoding: %@", error);

    if (beginning == NULL)
        return;

    XCTAssertEqual(beginning, [buffer mutableBytes], @"Size != buffer use");

    NSString* recovered = nil;

    error = nil;
    const uint8_t* end = kcder_decode_string(&recovered, &error, [buffer mutableBytes], [buffer mutableBytes] + size);

    XCTAssert(end != NULL, "Error decoding: %@", error);

    if (end == NULL)
        return;

    XCTAssertEqual(end, [buffer mutableBytes] + size, @"readback didn't use all the buffer");

    XCTAssertEqualObjects(string, recovered, @"Didn't get equal object");
    
}

- (void)testString {
    [self roundTripString: [NSString stringWithCString:"Test" encoding:NSUTF8StringEncoding]];
    [self roundTripString: [NSString stringWithCString:"Ã¼ðŸ˜ðŸ¸âœï¸â‚§âžœ" encoding:NSUTF8StringEncoding]];
}


@end
