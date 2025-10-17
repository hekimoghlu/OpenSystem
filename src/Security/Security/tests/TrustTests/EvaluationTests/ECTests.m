/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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
#include <Security/SecTrust.h>
#include <Security/SecCertificatePriv.h>
#include <Security/SecPolicy.h>
#include "OSX/utilities/SecCFWrappers.h"

#import "TrustEvaluationTestCase.h"
#import "ECTests_data.h"
#include "../TestMacroConversions.h"

/* Set this to 1 to test support for the legacy ecdsa-with-specified
 signature oid. */
#define TEST_ECDSA_WITH_SPECIFIED  0

@interface ECTests : TrustEvaluationTestCase
@end

#define trust_ok(CERT, ROOT, DATE, ...) \
({ \
XCTAssertTrue(test_trust_ok(CERT, sizeof(CERT), ROOT, sizeof(ROOT), DATE), __VA_ARGS__); \
})

static bool test_trust_ok(const uint8_t *cert_data, size_t cert_len,
                          const uint8_t *root_data, size_t root_len, const char *date_str) {
    SecTrustRef trust = NULL;
    SecPolicyRef policy = NULL;
    CFArrayRef anchors = NULL;
    SecCertificateRef cert = NULL, root = NULL;
    CFDateRef date = NULL;
    bool result = false;
    CFErrorRef error = NULL;
    require_string(cert = SecCertificateCreateWithBytes(NULL, cert_data, (CFIndex)cert_len),
                   errOut, "create cert");
    require_string(root = SecCertificateCreateWithBytes(NULL, root_data, (CFIndex)root_len),
                   errOut, "create root");
    
    policy = SecPolicyCreateSSL(false, NULL);
    require_noerr_string(SecTrustCreateWithCertificates(cert, policy, &trust),
                         errOut, "create trust with single cert");
    anchors = CFArrayCreate(NULL, (const void **)&root, 1,
                                       &kCFTypeArrayCallBacks);
    require_noerr_string(SecTrustSetAnchorCertificates(trust, anchors),
                         errOut, "set anchors");
    
    /* 2006/03/03 00:12:00 */
    date = CFDateCreate(NULL, 163037520.0);
    require_noerr_string(SecTrustSetVerifyDate(trust, date), errOut, "set date");
    result = SecTrustEvaluateWithError(trust, &error);
    
errOut:
    CFReleaseSafe(date);
    CFReleaseSafe(anchors);
    CFReleaseSafe(policy);
    CFReleaseSafe(root);
    CFReleaseSafe(cert);
    CFReleaseSafe(trust);
    CFReleaseSafe(error);
    return result;
}

@implementation ECTests

- (void)testMicrosoft_ECCerts {
    /* Verification of ECC certs created by Microsoft */
    trust_ok(RootP256_cer, RootP256_cer,
             "20060303001200", "RootP256_cer root");
#if TEST_ECDSA_WITH_SPECIFIED
    trust_ok(End_P256_Specified_SHA1_cer, RootP256_cer,
             "20060303001200", "End_P256_Specified_SHA1_cer");
    trust_ok(End_P256_Specified_SHA256_cer, RootP256_cer,
             "20060303001200", "End_P256_Specified_SHA256_cer");
    trust_ok(End_P384_Specified_SHA256_cer, RootP256_cer,
             "20060303001200", "End_P384_Specified_SHA256_cer");
    trust_ok(End_P384_Specified_SHA384_cer, RootP256_cer,
             "20060303001200", "End_P384_Specified_SHA384_cer");
    trust_ok(End_P521_Specified_SHA1_cer, RootP256_cer,
             "20060303001200", "End_P521_Specified_SHA1_cer");
#endif /* TEST_ECDSA_WITH_SPECIFIED */
    trust_ok(End_P256_combined_SHA256_cer, RootP256_cer,
             "20060303001200", "End_P256_combined_SHA256_cer");
    trust_ok(End_P384_combined_SHA256_cer, RootP256_cer,
             "20060303001200", "End_P384_combined_SHA256_cer");
    trust_ok(End_P384_combined_SHA1_cer, RootP256_cer,
             "20060303001200", "End_P384_combined_SHA1_cer");
    trust_ok(End_P521_combined_SHA1_cer, RootP256_cer,
             "20060303001200", "End_P521_combined_SHA1_cer");
    trust_ok(End_P256_combined_SHA512_cer, RootP256_cer,
             "20060303001200", "End_P256_combined_SHA512_cer");
    trust_ok(End_P521_combined_SHA512_cer, RootP256_cer,
             "20060303001200", "End_P521_combined_SHA512_cer");
}

- (void)testNSS_ECCerts {
    /* Verification of ECC certs created by NSS */
    trust_ok(ECCCA_cer, ECCCA_cer,
             "20060303001200", "ECCCA_cer root");
    trust_ok(ECCp192_cer, ECCCA_cer,
             "20060303001200", "ECCp192_cer");
    trust_ok(ECCp256_cer, ECCCA_cer,
             "20060303001200", "ECCp256_cer");
    trust_ok(ECCp384_cer, ECCCA_cer,
             "20060303001200", "ECCp384_cer");
    trust_ok(ECCp521_cer, ECCCA_cer,
             "20060303001200", "ECCp521_cer");
}

- (void)testOpenSSL_ECCerts {
    /* Verification of ECC certs created by OpenSSL */
    trust_ok(secp256r1ca_cer, secp256r1ca_cer,
             "20060303001200", "secp256r1ca_cer root");
    trust_ok(secp256r1server_secp256r1ca_cer, secp256r1ca_cer,
             "20060303001200", "secp256r1server_secp256r1ca_cer");
    trust_ok(secp384r1ca_cer, secp384r1ca_cer,
             "20060303001200", "secp384r1ca_cer root");
    trust_ok(secp384r1server_secp384r1ca_cer, secp384r1ca_cer,
             "20060303001200", "secp384r1server_secp384r1ca_cer");
    trust_ok(secp521r1ca_cer, secp521r1ca_cer,
             "20060303001200", "secp521r1ca_cer root");
    trust_ok(secp521r1server_secp521r1ca_cer, secp521r1ca_cer,
             "20060303001200", "secp521r1server_secp521r1ca_cer");
}

@end
