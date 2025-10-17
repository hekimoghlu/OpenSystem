/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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
//  SecXPCHelperTests.m
//  SecurityUtilityTests
//

#import <XCTest/XCTest.h>
#import "utilities/SecXPCHelper.h"

@interface UnsafeType : NSObject <NSCopying>
@property(class, readonly) BOOL supportsSecureCoding;
- (id)initWithCoder:(NSCoder *)decoder;
@end

@implementation UnsafeType

- (id)init {
    return [super init];
}

- (NSString *)description {
    return @"UnsafeType";
}

- (id)copyWithZone:(nullable NSZone *)zone {
    return [[UnsafeType alloc] init];
}

+ (BOOL)supportsSecureCoding {
    return NO;
}

- (id)initWithCoder:(NSCoder *)decoder {
    return [[UnsafeType alloc] init];
}

@end

@interface SafeType : NSObject
@property(class, readonly) BOOL supportsSecureCoding;
- (id)initWithCoder:(NSCoder *)decoder;
@end

@implementation SafeType

- (id)init {
    return [super init];
}

+ (BOOL)supportsSecureCoding {
    return YES;
}

- (id)initWithCoder:(NSCoder *)decoder {
    return [[SafeType alloc] init];
}

@end

@interface SecXPCHelperTests : XCTestCase
@end

@implementation SecXPCHelperTests

- (void)testSanitizerWithNilError {
    XCTAssertNil([SecXPCHelper cleanseErrorForXPC:nil]);
}

- (void)testSanitizerWithCleanError {
    NSDictionary *cleanInfo = @{@"Key" : @"Safe String"};
    NSError *error = [NSError errorWithDomain:NSOSStatusErrorDomain code:-1 userInfo:cleanInfo];
    NSError *cleansed = [SecXPCHelper cleanseErrorForXPC:error];
    XCTAssertNotNil(cleansed);
    XCTAssertTrue(cleansed.code == error.code);
    XCTAssertTrue(cleansed.domain == error.domain);
    XCTAssertTrue([cleansed.userInfo isEqualToDictionary:cleanInfo]);
}

- (void)testSanitizerWithSafeCodingError {
    SafeType *safe = [[SafeType alloc] init];
    NSDictionary *cleanInfo = @{@"Key" : safe};
    NSDictionary *sanitizedInfo = @{@"Key" : [[safe class] description]};
    NSError *error = [NSError errorWithDomain:NSOSStatusErrorDomain code:-1 userInfo:cleanInfo];
    NSError *cleansed = [SecXPCHelper cleanseErrorForXPC:error];
    XCTAssertNotNil(cleansed);
    XCTAssertTrue(cleansed.code == error.code);
    XCTAssertTrue(cleansed.domain == error.domain);
    XCTAssertTrue([cleansed.userInfo isEqualToDictionary:sanitizedInfo]);
}

- (void)testSanitizerWithDirtyUnsafeError {
    UnsafeType *unsafe = [[UnsafeType alloc] init];
    NSDictionary *cleanInfo = @{@"Key" : unsafe};
    NSDictionary *sanitizedInfo = @{@"Key" : [[unsafe class] description]};
    NSError *error = [NSError errorWithDomain:NSOSStatusErrorDomain code:-1 userInfo:cleanInfo];
    NSError *cleansed = [SecXPCHelper cleanseErrorForXPC:error];
    XCTAssertNotNil(cleansed);
    XCTAssertTrue(cleansed.code == error.code);
    XCTAssertTrue(cleansed.domain == error.domain);
    XCTAssertTrue([cleansed.userInfo isEqualToDictionary:sanitizedInfo]);
}

- (void)testSanitizerWithDirtyUnsafeKey {
    UnsafeType *unsafe = [[UnsafeType alloc] init];
    NSDictionary *unsafeInfo = @{ unsafe: @"value"};
    NSError *error = [NSError errorWithDomain:NSOSStatusErrorDomain code:-1 userInfo:unsafeInfo];
    NSError *cleansed = [SecXPCHelper cleanseErrorForXPC:error];
    XCTAssertNotNil(cleansed);
    XCTAssertTrue(cleansed.code == error.code);
    XCTAssertTrue(cleansed.domain == error.domain);
    XCTAssertEqualObjects(cleansed.userInfo, @{@"UnsafeType": @"value"});
}


- (void)testErrorEncodingUnsafe {
    NSDictionary *unsafeInfo = @{ @"info" : [[NSObject alloc] init] };
    NSError *unsafeError = [NSError errorWithDomain:@"domain" code:23 userInfo:unsafeInfo];
    NSData *unsafeEncodedData = nil;
    bool exceptionCaught = false;
    @try {
        unsafeEncodedData = [SecXPCHelper encodedDataFromError:unsafeError];
    } @catch (NSException *e) {
        XCTAssertNotNil(e);
        XCTAssertEqualObjects(e.name, NSInvalidUnarchiveOperationException);
        exceptionCaught = true;
    }
    XCTAssertTrue(exceptionCaught);
    XCTAssertNil(unsafeEncodedData);
}

- (void)testErrorEncodingSafe {
    NSDictionary *safeInfo = @{ @"info" : @(57) };
    NSError *safeError = [NSError errorWithDomain:@"domain" code:19 userInfo:safeInfo];
    NSData *safeEncodedData = [SecXPCHelper encodedDataFromError:safeError];
    XCTAssertNotNil(safeEncodedData);
    NSError *decodedSafeError = [SecXPCHelper errorFromEncodedData:safeEncodedData];
    XCTAssertNotNil(decodedSafeError);
    XCTAssertEqualObjects(decodedSafeError.domain, safeError.domain);
    XCTAssertEqual(decodedSafeError.code, safeError.code);
    XCTAssertEqualObjects(decodedSafeError.userInfo, safeError.userInfo);

    /* Double-check the archive key */
    NSKeyedUnarchiver *unarchiver = [[NSKeyedUnarchiver alloc] initForReadingFromData:safeEncodedData error:NULL];
    XCTAssertNotNil(unarchiver);
    XCTAssertTrue([unarchiver containsValueForKey:@"error"], "missing expected key (this is a wire format!)");
}

@end
