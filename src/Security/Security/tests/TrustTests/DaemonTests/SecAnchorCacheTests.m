/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 6, 2023.
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
//  SecAnchorCacheTests.m
//  Security
//
//

#import <Foundation/Foundation.h>
#include <AssertMacros.h>
#import <XCTest/XCTest.h>

#import <Security/SecPolicyPriv.h>
#include <utilities/SecAppleAnchorPriv.h>
#import "trust/trustd/SecAnchorCache.h"

#import "TrustDaemonTestCase.h"

@interface AnchorCacheTests : TrustDaemonTestCase
@end

@implementation AnchorCacheTests

+ (void)setUp {
    [super setUp];
    SecAnchorCacheInitialize();
}

- (void)testCopyAnchors {
#if TARGET_OS_BRIDGE
    /* bridgeOS doesn't use trust store */
    XCTSkip();
#endif
    /* Apple Anchors */
    NSArray *anchors = CFBridgingRelease(SecAnchorCacheCopyAnchors(kSecPolicyAppleMobileAsset));
    XCTAssertNotNil(anchors);
    NSArray *appleAnchors = (__bridge NSArray *)SecGetAppleTrustAnchors(false);
    XCTAssertEqualObjects(anchors, appleAnchors);

    /* Constrained Anchors */
    anchors = CFBridgingRelease(SecAnchorCacheCopyAnchors(kSecPolicyAppleVerifiedMark));
    XCTAssertNotNil(anchors);
    XCTAssertGreaterThan(anchors.count, 0);

    anchors = CFBridgingRelease(SecAnchorCacheCopyAnchors(kSecPolicyAppleMDLTerminalAuth));
    XCTAssertNotNil(anchors);
    XCTAssertGreaterThan(anchors.count, 0);

    /* System anchors */
    anchors = CFBridgingRelease(SecAnchorCacheCopyAnchors(kSecPolicyAppleX509Basic));
    XCTAssertNotNil(anchors);
    XCTAssertGreaterThan(anchors.count, 1);

    NSArray *ocspAnchors = CFBridgingRelease(SecAnchorCacheCopyAnchors(kSecPolicyAppleOCSPSigner));
    XCTAssertNotNil(ocspAnchors);
    XCTAssertEqualObjects(anchors, ocspAnchors);

    /* Prime anchor cache and then copy anchors again */
    CFStringRef sectigoServerAuthRootLookupKey = CFSTR("6FAEB525494DCEC35FC629C946482912999E2EC8"); // Sectigo Public Server Authentication Root R46
    NSArray *parents = CFBridgingRelease(SecAnchorCacheCopyParentCertificates(sectigoServerAuthRootLookupKey));
    XCTAssertNotNil(parents);
    NSArray *cachedAnchors = CFBridgingRelease(SecAnchorCacheCopyAnchors(kSecPolicyAppleX509Basic));
    XCTAssertNotNil(cachedAnchors);
    XCTAssertEqualObjects(anchors, cachedAnchors);
}

@end
