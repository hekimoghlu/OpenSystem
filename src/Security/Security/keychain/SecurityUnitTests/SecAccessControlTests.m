/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 18, 2023.
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
#import <Security/SecAccessControlPriv.h>


@interface SecAccessControlTests : XCTestCase

@end

@implementation SecAccessControlTests

- (void)testExplicitBind {
    NSError *error;
    id sac = CFBridgingRelease(SecAccessControlCreate(kCFAllocatorDefault, (void *)&error));
    XCTAssertNotNil(sac, @"Failed to create an empty ACL: %@", error);
    XCTAssert(SecAccessControlAddConstraintForOperation((__bridge SecAccessControlRef)sac, CFSTR("op"), (__bridge SecAccessConstraintRef)@{}, (void *)&error), @"Failed to add dict constraint: %@", error);

    XCTAssertFalse(SecAccessControlIsBound((__bridge SecAccessControlRef)sac), @"ACL is not expected to be bound");
    SecAccessControlSetBound((__bridge SecAccessControlRef)sac, true);
    XCTAssertTrue(SecAccessControlIsBound((__bridge SecAccessControlRef)sac), @"ACL is expected to be bound");
    SecAccessControlSetBound((__bridge SecAccessControlRef)sac, false);
    XCTAssertFalse(SecAccessControlIsBound((__bridge SecAccessControlRef)sac), @"ACL is not expected to be bound");
}

- (void)testImplicitBound {
    @autoreleasepool {
        NSError *error;
        id sac = CFBridgingRelease(SecAccessControlCreate(kCFAllocatorDefault, (void *)&error));
        XCTAssertNotNil(sac, @"Failed to create an empty ACL: %@", error);

        XCTAssertTrue(SecAccessControlIsBound((__bridge SecAccessControlRef)sac), @"%@ is expected to be implicitly bound", sac);
    }

    @autoreleasepool {
        NSError *error;
        id sac = CFBridgingRelease(SecAccessControlCreateWithFlags(kCFAllocatorDefault, kSecAttrAccessibleAfterFirstUnlock, 0, (void *)&error));
        XCTAssertNotNil(sac, @"Failed to create ACL: %@", error);

        XCTAssertTrue(SecAccessControlIsBound((__bridge SecAccessControlRef)sac), @"%@ is expected to be implicitly bound", sac);
    }

    @autoreleasepool {
        NSError *error;
        id sac = CFBridgingRelease(SecAccessControlCreateWithFlags(kCFAllocatorDefault, kSecAttrAccessibleAfterFirstUnlock, kSecAccessControlPrivateKeyUsage, (void *)&error));
        XCTAssertNotNil(sac, @"Failed to create ACL: %@", error);

        XCTAssertTrue(SecAccessControlIsBound((__bridge SecAccessControlRef)sac), @"%@ is expected to be implicitly bound", sac);
    }

    @autoreleasepool {
        NSError *error;
        id sac = CFBridgingRelease(SecAccessControlCreateWithFlags(kCFAllocatorDefault, kSecAttrAccessibleAfterFirstUnlock, kSecAccessControlApplicationPassword, (void *)&error));
        XCTAssertNotNil(sac, @"Failed to create ACL: %@", error);

        XCTAssertFalse(SecAccessControlIsBound((__bridge SecAccessControlRef)sac), @"%@ is not expected to be implicitly bound", sac);
    }

    @autoreleasepool {
        NSError *error;
        id sac = CFBridgingRelease(SecAccessControlCreateWithFlags(kCFAllocatorDefault, kSecAttrAccessibleAfterFirstUnlock, kSecAccessControlPrivateKeyUsage | kSecAccessControlApplicationPassword, (void *)&error));
        XCTAssertNotNil(sac, @"Failed to create ACL: %@", error);

        XCTAssertFalse(SecAccessControlIsBound((__bridge SecAccessControlRef)sac), @"%@ is not expected to be implicitly bound", sac);
    }

    @autoreleasepool {
        NSError *error;
        id sac = CFBridgingRelease(SecAccessControlCreateWithFlags(kCFAllocatorDefault, kSecAttrAccessibleAfterFirstUnlock, kSecAccessControlBiometryAny, (void *)&error));
        XCTAssertNotNil(sac, @"Failed to create ACL: %@", error);

        XCTAssertFalse(SecAccessControlIsBound((__bridge SecAccessControlRef)sac), @"%@ is not expected to be implicitly bound", sac);
    }

    @autoreleasepool {
        NSError *error;
        id sac = CFBridgingRelease(SecAccessControlCreateWithFlags(kCFAllocatorDefault, kSecAttrAccessibleAfterFirstUnlock, kSecAccessControlPrivateKeyUsage | kSecAccessControlBiometryAny, (void *)&error));
        XCTAssertNotNil(sac, @"Failed to create ACL: %@", error);

        XCTAssertFalse(SecAccessControlIsBound((__bridge SecAccessControlRef)sac), @"%@ is not expected to be implicitly bound", sac);
    }

    @autoreleasepool {
        NSError *error;
        id sac = CFBridgingRelease(SecAccessControlCreateWithFlags(kCFAllocatorDefault, kSecAttrAccessibleAfterFirstUnlock, kSecAccessControlUserPresence, (void *)&error));
        XCTAssertNotNil(sac, @"Failed to create ACL: %@", error);

        XCTAssertFalse(SecAccessControlIsBound((__bridge SecAccessControlRef)sac), @"%@ is not expected to be implicitly bound", sac);
    }

    @autoreleasepool {
        NSError *error;
        id sac = CFBridgingRelease(SecAccessControlCreateWithFlags(kCFAllocatorDefault, kSecAttrAccessibleAfterFirstUnlock, kSecAccessControlPrivateKeyUsage | kSecAccessControlUserPresence, (void *)&error));
        XCTAssertNotNil(sac, @"Failed to create ACL: %@", error);

        XCTAssertFalse(SecAccessControlIsBound((__bridge SecAccessControlRef)sac), @"%@ is not expected to be implicitly bound", sac);
    }
}

@end
