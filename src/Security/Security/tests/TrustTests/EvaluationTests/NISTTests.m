/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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
#include <AssertMacros.h>
#import <XCTest/XCTest.h>
#import <Foundation/Foundation.h>

#include <Security/SecCertificate.h>
#include <Security/SecCertificatePriv.h>
#include <Security/SecPolicyPriv.h>
#include <Security/SecTrustPriv.h>
#include <utilities/SecCFWrappers.h>

#import "TrustEvaluationTestCase.h"

@interface NISTTests : TrustEvaluationTestCase
@end

@implementation NISTTests

- (void)testPKITSCerts {
    SecPolicyRef basicPolicy = SecPolicyCreateBasicX509();
    NSDate *testDate = CFBridgingRelease(CFDateCreateForGregorianZuluDay(NULL, 2011, 9, 1));

    /* Run the tests. */
    [self runCertificateTestForDirectory:basicPolicy subDirectory:@"nist-certs" verifyDate:testDate];

    CFReleaseSafe(basicPolicy);
}

- (void)testNoBasicConstraintsAnchor_UserTrusted {
#if TARGET_OS_BRIDGE // bridgeOS doesn't have trust settings
    XCTSkip();
#endif
    SecCertificateRef leaf = (__bridge SecCertificateRef)[self SecCertificateCreateFromResource:@"InvalidMissingbasicConstraintsTest1EE"
                                                                                   subdirectory:@"nist-certs"];
    SecCertificateRef ca = (__bridge SecCertificateRef)[self SecCertificateCreateFromResource:@"MissingbasicConstraintsCACert"
                                                                                 subdirectory:@"nist-certs"];
    SecTrustRef trust = NULL;
    NSArray *certs = @[(__bridge id)leaf, (__bridge id)ca];

    XCTAssertEqual(errSecSuccess, SecTrustCreateWithCertificates((__bridge CFArrayRef)certs, NULL, &trust));
    NSDate *testDate = CFBridgingRelease(CFDateCreateForGregorianZuluDay(NULL, 2011, 9, 1));
    XCTAssertEqual(errSecSuccess, SecTrustSetVerifyDate(trust, (__bridge CFDateRef)testDate));

    id persistentRef = [self addTrustSettingsForCert:ca];
    CFErrorRef error = nil;
    XCTAssertFalse(SecTrustEvaluateWithError(trust, &error));
    XCTAssertNotEqual(error, NULL);
    if (error) {
        XCTAssertEqual(CFErrorGetCode(error), errSecNoBasicConstraints);
    }

    [self removeTrustSettingsForCert:ca persistentRef:persistentRef];
    CFReleaseNull(leaf);
    CFReleaseNull(ca);
    CFReleaseNull(error);
}

- (void)testNoBasicConstraintsAnchor_AppTrusted {
    SecCertificateRef leaf = (__bridge SecCertificateRef)[self SecCertificateCreateFromResource:@"InvalidMissingbasicConstraintsTest1EE"
                                                                                   subdirectory:@"nist-certs"];
    SecCertificateRef ca = (__bridge SecCertificateRef)[self SecCertificateCreateFromResource:@"MissingbasicConstraintsCACert"
                                                                                 subdirectory:@"nist-certs"];
    SecTrustRef trust = NULL;
    NSArray *certs = @[(__bridge id)leaf, (__bridge id)ca];
    NSArray *anchor = @[(__bridge id)ca];

    XCTAssertEqual(errSecSuccess, SecTrustCreateWithCertificates((__bridge CFArrayRef)certs, NULL, &trust));
    NSDate *testDate = CFBridgingRelease(CFDateCreateForGregorianZuluDay(NULL, 2011, 9, 1));
    XCTAssertEqual(errSecSuccess, SecTrustSetVerifyDate(trust, (__bridge CFDateRef)testDate));
    XCTAssertEqual(errSecSuccess, SecTrustSetAnchorCertificates(trust, (__bridge CFArrayRef)anchor));

    CFErrorRef error = nil;
    XCTAssertFalse(SecTrustEvaluateWithError(trust, &error));
    XCTAssertNotEqual(error, NULL);
    if (error) {
        XCTAssertEqual(CFErrorGetCode(error), errSecNoBasicConstraints);
    }

    CFReleaseNull(leaf);
    CFReleaseNull(ca);
    CFReleaseNull(error);
}

- (void)testNotCABasicConstraintsAnchor_UserTrusted {
#if TARGET_OS_BRIDGE // bridgeOS doesn't have trust settings
    XCTSkip();
#endif
    SecCertificateRef leaf = (__bridge SecCertificateRef)[self SecCertificateCreateFromResource:@"InvalidcAFalseTest2EE"
                                                                                   subdirectory:@"nist-certs"];
    SecCertificateRef ca = (__bridge SecCertificateRef)[self SecCertificateCreateFromResource:@"basicConstraintsCriticalcAFalseCACert"
                                                                                 subdirectory:@"nist-certs"];
    SecTrustRef trust = NULL;
    NSArray *certs = @[(__bridge id)leaf, (__bridge id)ca];

    XCTAssertEqual(errSecSuccess, SecTrustCreateWithCertificates((__bridge CFArrayRef)certs, NULL, &trust));
    NSDate *testDate = CFBridgingRelease(CFDateCreateForGregorianZuluDay(NULL, 2011, 9, 1));
    XCTAssertEqual(errSecSuccess, SecTrustSetVerifyDate(trust, (__bridge CFDateRef)testDate));

    id persistentRef = [self addTrustSettingsForCert:ca];
    CFErrorRef error = nil;
    XCTAssertFalse(SecTrustEvaluateWithError(trust, &error));
    XCTAssertNotEqual(error, NULL);
    if (error) {
        XCTAssertEqual(CFErrorGetCode(error), errSecNoBasicConstraintsCA);
    }

    [self removeTrustSettingsForCert:ca persistentRef:persistentRef];
    CFReleaseNull(leaf);
    CFReleaseNull(ca);
    CFReleaseNull(error);
}

- (void)testNotCABasicConstraintsAnchor_AppTrusted {
    SecCertificateRef leaf = (__bridge SecCertificateRef)[self SecCertificateCreateFromResource:@"InvalidcAFalseTest2EE"
                                                                                   subdirectory:@"nist-certs"];
    SecCertificateRef ca = (__bridge SecCertificateRef)[self SecCertificateCreateFromResource:@"basicConstraintsCriticalcAFalseCACert"
                                                                                 subdirectory:@"nist-certs"];
    SecTrustRef trust = NULL;
    NSArray *certs = @[(__bridge id)leaf, (__bridge id)ca];
    NSArray *anchor = @[(__bridge id)ca];

    XCTAssertEqual(errSecSuccess, SecTrustCreateWithCertificates((__bridge CFArrayRef)certs, NULL, &trust));
    NSDate *testDate = CFBridgingRelease(CFDateCreateForGregorianZuluDay(NULL, 2011, 9, 1));
    XCTAssertEqual(errSecSuccess, SecTrustSetVerifyDate(trust, (__bridge CFDateRef)testDate));
    XCTAssertEqual(errSecSuccess, SecTrustSetAnchorCertificates(trust, (__bridge CFArrayRef)anchor));

    CFErrorRef error = nil;
    XCTAssertFalse(SecTrustEvaluateWithError(trust, &error));
    XCTAssertNotEqual(error, NULL);
    if (error) {
        XCTAssertEqual(CFErrorGetCode(error), errSecNoBasicConstraintsCA);
    }

    CFReleaseNull(leaf);
    CFReleaseNull(ca);
    CFReleaseNull(error);
}

@end
