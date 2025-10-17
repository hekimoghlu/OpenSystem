/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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
#include "OSX/utilities/SecCFWrappers.h"
#include <Security/SecCertificatePriv.h>
#include <Security/SecTrustSettings.h>
#include <Security/SecTrustSettingsPriv.h>
#include <Security/SecTrust.h>
#include <Security/SecTrustPriv.h>
#include <Security/SecFramework.h>
#include <Security/SecTrustStore.h>
#include <Security/SecPolicyPriv.h>

#include "../TestMacroConversions.h"
#include "TrustFrameworkTestCase.h"
#include "TrustSettingsInterfaceTests_data.h"

@interface TrustSettingsInterfaceTests : TrustFrameworkTestCase
@end

@implementation TrustSettingsInterfaceTests

#if TARGET_OS_OSX
- (void)testCopySystemAnchors {
    CFArrayRef certArray;
    ok_status(SecTrustCopyAnchorCertificates(&certArray), "copy anchors");
    SecCertificateRef qwac_anchor = SecCertificateCreateWithBytes(NULL, _harica_qwac_anchor, sizeof(_harica_qwac_anchor));
    XCTAssertFalse(CFArrayContainsValue(certArray, CFRangeMake(0, CFArrayGetCount(certArray)), qwac_anchor));
    CFReleaseSafe(certArray);

    ok_status(SecTrustSettingsCopyCertificates(kSecTrustSettingsDomainSystem, &certArray), "copy certificates");
    XCTAssertFalse(CFArrayContainsValue(certArray, CFRangeMake(0, CFArrayGetCount(certArray)), qwac_anchor));
    CFReleaseSafe(certArray);
    CFReleaseNull(qwac_anchor);
}
#endif

- (void)testTrustStoreCopyAnchors {
#if TARGET_OS_BRIDGE
    XCTSkip();
#endif
    /* Repeat the evaluation test for the CopyAnchors test with a live trustd (testing that the XPC flow is functional) */
    SecTrustStoreRef systemTS = SecTrustStoreForDomain(kSecTrustStoreDomainSystem);

    CFArrayRef unconstrainedStore = NULL;
    XCTAssertEqual(errSecSuccess, SecTrustStoreCopyAll(systemTS, &unconstrainedStore));
    XCTAssertNotEqual(unconstrainedStore, NULL);

    // Unconstrained system trust store
    NSArray *anchors = CFBridgingRelease(SecTrustStoreCopyAnchors(systemTS, kSecPolicyAppleX509Basic));
    XCTAssertNotNil(anchors);
    XCTAssertEqualObjects(anchors, (__bridge NSArray*)unconstrainedStore);

    // Apple anchors
    anchors = CFBridgingRelease(SecTrustStoreCopyAnchors(systemTS, kSecPolicyAppleiPhoneApplicationSigning));
    XCTAssertNotNil(anchors);
    XCTAssertGreaterThanOrEqual(anchors.count, 3);

    // Constrained anchor policy
    anchors = CFBridgingRelease(SecTrustStoreCopyAnchors(systemTS, kSecPolicyAppleMDLTerminalAuth));
    XCTAssertNotNil(anchors);
    XCTAssertGreaterThanOrEqual(anchors.count, 1);

    // Currently no constrained anchors defined
    anchors = CFBridgingRelease(SecTrustStoreCopyAnchors(systemTS, kSecPolicyAppleiAP));
    XCTAssertGreaterThanOrEqual(anchors.count, 0);
}

- (void)testSetCTExceptions {
#if TARGET_OS_BRIDGE
    XCTSkip();
#endif
    CFErrorRef error = NULL;
    const CFStringRef TrustTestsAppID = CFSTR("com.apple.trusttests");
    CFDictionaryRef copiedExceptions = NULL;

    /* Verify no exceptions set */
    is(copiedExceptions = SecTrustStoreCopyCTExceptions(NULL, NULL), NULL, "no exceptions set");
    if (copiedExceptions) {
        /* If we're starting out with exceptions set, a lot of the following will also fail, so just skip them */
        CFReleaseNull(copiedExceptions);
        return;
    }

    /* Set exceptions with specified AppID */
    NSDictionary *exceptions1 = @{
                                  (__bridge NSString*)kSecCTExceptionsDomainsKey: @[@"test.apple.com", @".test.apple.com"],
                                  };
    ok(SecTrustStoreSetCTExceptions(TrustTestsAppID, (__bridge CFDictionaryRef)exceptions1, &error),
       "failed to set exceptions for SecurityTests: %@", error);

    /* Copy all exceptions (with only one set) */
    ok(copiedExceptions = SecTrustStoreCopyCTExceptions(NULL, &error),
       "failed to copy all exceptions: %@", error);
    ok([exceptions1 isEqualToDictionary:(__bridge NSDictionary*)copiedExceptions],
       "got the wrong exceptions back");
    CFReleaseNull(copiedExceptions);

    /* Copy this app's exceptions */
    ok(copiedExceptions = SecTrustStoreCopyCTExceptions(TrustTestsAppID, &error),
       "failed to copy SecurityTests' exceptions: %@", error);
    ok([exceptions1 isEqualToDictionary:(__bridge NSDictionary*)copiedExceptions],
       "got the wrong exceptions back");
    CFReleaseNull(copiedExceptions);

    /* Set different exceptions with implied AppID */
    NSDictionary *exceptions2 = @{
                                  (__bridge NSString*)kSecCTExceptionsDomainsKey: @[@".test.apple.com"],
                                  };
    ok(SecTrustStoreSetCTExceptions(NULL, (__bridge CFDictionaryRef)exceptions2, &error),
       "failed to set exceptions for this app: %@", error);

    /* Ensure exceptions are replaced for SecurityTests */
    ok(copiedExceptions = SecTrustStoreCopyCTExceptions(TrustTestsAppID, &error),
       "failed to copy SecurityTests' exceptions: %@", error);
    ok([exceptions2 isEqualToDictionary:(__bridge NSDictionary*)copiedExceptions],
       "got the wrong exceptions back");
    CFReleaseNull(copiedExceptions);

    /* Set exceptions with bad inputs */
    NSDictionary *badExceptions = @{
                                    (__bridge NSString*)kSecCTExceptionsDomainsKey: @[@"test.apple.com", @".test.apple.com"],
                                    @"not a key": @"not a value",
                                    };
    is(SecTrustStoreSetCTExceptions(NULL, (__bridge CFDictionaryRef)badExceptions, &error), false,
       "set exceptions with unknown key");
    if (error) {
        is(CFErrorGetCode(error), errSecParam, "bad input produced unxpected error code: %ld", (long)CFErrorGetCode(error));
    } else {
        fail("expected failure to set NULL exceptions");
    }
    CFReleaseNull(error);

    /* Remove exceptions */
    ok(SecTrustStoreSetCTExceptions(NULL, NULL, &error),
       "failed to set empty array exceptions for this app: %@", error);
    is(copiedExceptions = SecTrustStoreCopyCTExceptions(NULL, NULL), NULL, "no exceptions set");
}

- (NSData *)random
{
    uint8_t random[32];
    (void)SecRandomCopyBytes(kSecRandomDefault, sizeof(random), random);
    return [[NSData alloc] initWithBytes:random length:sizeof(random)];
}

- (void)testSetTransparentConnections {
#if TARGET_OS_BRIDGE
    XCTSkip();
#endif
    CFErrorRef error = NULL;
    const CFStringRef TrustTestsAppID = CFSTR("com.apple.trusttests");
    CFArrayRef copiedPins = NULL;

    /* Verify no pins set */
    copiedPins = SecTrustStoreCopyTransparentConnectionPins(NULL, NULL);
    XCTAssertEqual(copiedPins, NULL);
    if (copiedPins) {
        /* If we're startign out with pins set, a lot of the following will also fail, so just skip them */
        CFReleaseNull(copiedPins);
        return;
    }

    /* Set pin with specified AppID */
    NSArray *pin1 = @[@{
        (__bridge NSString*)kSecTrustStoreHashAlgorithmKey : @"sha256",
        (__bridge NSString*)kSecTrustStoreSPKIHashKey : [self random]
    }];
    /* Set pin with specified AppID */
    XCTAssert(SecTrustStoreSetTransparentConnectionPins(TrustTestsAppID, (__bridge CFArrayRef)pin1, &error),
              "failed to set pins: %@", error);

    /* Copy all pins (with only one set) */
    XCTAssertNotEqual(NULL, copiedPins = SecTrustStoreCopyTransparentConnectionPins(NULL, &error),
                      "failed to copy all pins: %@", error);
    XCTAssertEqualObjects(pin1, (__bridge NSArray*)copiedPins);
    CFReleaseNull(copiedPins);

    /* Copy this app's pins */
    XCTAssertNotEqual(NULL, copiedPins = SecTrustStoreCopyTransparentConnectionPins(TrustTestsAppID, &error),
                      "failed to copy this app's pins: %@", error);
    XCTAssertEqualObjects(pin1, (__bridge NSArray*)copiedPins);
    CFReleaseNull(copiedPins);

    /* Set a different pin with implied AppID and ensure pins are replaced */
    NSArray *pin2 = @[@{
        (__bridge NSString*)kSecTrustStoreHashAlgorithmKey : @"sha256",
        (__bridge NSString*)kSecTrustStoreSPKIHashKey : [self random]
    }];
    XCTAssert(SecTrustStoreSetTransparentConnectionPins(NULL, (__bridge CFArrayRef)pin2, &error),
              "failed to set pins: %@", error);
    XCTAssertNotEqual(NULL, copiedPins = SecTrustStoreCopyTransparentConnectionPins(TrustTestsAppID, &error),
                      "failed to copy this app's pins: %@", error);
    XCTAssertEqualObjects(pin2, (__bridge NSArray*)copiedPins);
    CFReleaseNull(copiedPins);

    /* Set exceptions with bad inputs */
    NSArray *badPins = @[@{
         (__bridge NSString*)kSecTrustStoreHashAlgorithmKey : @"sha256",
         @"not a key" : @"not a value"
    }];
    XCTAssertFalse(SecTrustStoreSetTransparentConnectionPins(NULL, (__bridge CFArrayRef)badPins, &error));
    if (error) {
        is(CFErrorGetCode(error), errSecParam, "bad input produced unxpected error code: %ld", (long)CFErrorGetCode(error));
    } else {
        fail("expected failure to set NULL pins");
    }
    CFReleaseNull(error);

    /* Reset remaining pins */
    XCTAssert(SecTrustStoreSetTransparentConnectionPins(TrustTestsAppID, NULL, &error),
              "failed to reset pins: %@", error);
    XCTAssertEqual(NULL, copiedPins = SecTrustStoreCopyTransparentConnectionPins(NULL, &error),
                   "failed to copy all pins: %@", error);
    CFReleaseNull(copiedPins);
}

- (void)testTrustStoreRemoveAll {
#if TARGET_OS_BRIDGE
    XCTSkip();
#else
    XCTAssertEqual(errSecSuccess, SecTrustStoreRemoveAll(SecTrustStoreForDomain(kSecTrustStoreDomainUser)));
#endif
}

- (void)testTrustStoreAssetVersion {
#if TARGET_OS_BRIDGE
    XCTSkip();
#else
    NSString *tsAssetVersion = (__bridge_transfer NSString *)SecTrustCopyTrustStoreAssetVersion(NULL);
    if (!tsAssetVersion) {
        XCTSkip(); /* asset currently unavailable */
    } else {
        NSCharacterSet *asciiDigits = [NSCharacterSet characterSetWithRange:NSMakeRange('0', '9')];
        NSMutableString *testString = [NSMutableString stringWithString:tsAssetVersion];
        while (true) {
            NSRange range = [testString rangeOfCharacterFromSet:asciiDigits];
            if (range.location == NSNotFound || range.length <= 0) {
                break;
            }
            [testString replaceCharactersInRange:range withString:@""];
        }
        /* valid asset version format looks like "1.0.0.11.1234,0", where the actual numbers vary.
           If we remove all digits, we expect the string "....," as the result. */
        XCTAssertEqualObjects(testString, @"....,", "expected remainder after removing digits");
    }
#endif
}

@end
