/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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
#include <Security/SecPolicyPriv.h>
#include <Security/SecPolicyInternal.h>
#include <utilities/array_size.h>
#include <utilities/SecCFWrappers.h>
#include <Foundation/NSJSONSerialization.h>
#include "SecJWS.h"
#include "TrustFrameworkTestCase.h"

@interface SecJWSEncoder (Private)
- (BOOL) appendPaddedToData:(NSMutableData *)data ptr:(const uint8_t *)ptr len:(size_t)len expected:(size_t)expLen;
@end

@interface CertificateRequestTests : TrustFrameworkTestCase
@end

@implementation CertificateRequestTests

static void test_jws_signature(void) {
    // create an encoder, which generates a default signing key pair
    SecJWSEncoder *encoder = [[SecJWSEncoder alloc] init];
    NSMutableDictionary *payload = [NSMutableDictionary dictionary];
    payload[@"termsOfServiceAgreed"] = @(YES); // fixed payload item
    NSString *url = @"https://localhost:14000/dir"; // fixed url
    NSString *nonce = [[NSUUID UUID] UUIDString]; // one-time nonce
    NSError *encodeError = NULL;
    NSString *jwsString = [encoder encodedJWSWithPayload:payload kid:nil nonce:nonce url:url error:&encodeError];
    // verify we successfully encoded and signed the JSON
    XCTAssertNotEqual(jwsString, NULL);

    // convert the JSON into a dictionary
    NSError *decodeError = NULL;
    NSData *jwsData = [jwsString dataUsingEncoding:NSUTF8StringEncoding];
    XCTAssertNotEqual(jwsData, NULL);
    NSDictionary *jwsDict = [NSJSONSerialization JSONObjectWithData:jwsData options:0 error:&decodeError];
    XCTAssertNotEqual(jwsDict, NULL);

    // verify we can make a compact encoded string from its components
    NSString *jwsCompactString = [NSString stringWithFormat:@"%@.%@.%@", jwsDict[@"protected"], jwsDict[@"payload"], jwsDict[@"signature"]];
    XCTAssertNotEqual(jwsCompactString, NULL);

    // test that the compact encoded string decodes successfully
    SecJWSDecoder *decoder = [[SecJWSDecoder alloc] initWithJWSCompactEncodedString:jwsCompactString keyID:nil publicKey:encoder.publicKey];
    // %%% check verificationError when SecJWSDecoder has been completed
    // XCTAssertEqual(decoder.verificationError, NULL);

    // test that the signature is the expected size (rdar://116576444)
    XCTAssertEqual([decoder.signature length], 64);
}

- (void)testJWSSignatures {
    // generate and verify JWS signatures
    // (each invocation uses a different key and nonce)
    CFIndex count = 8; // number of signatures to generate
    for (CFIndex i = 0; i < count; i++) {
        test_jws_signature();
    }
}

- (void)testZeroPadding {
    SecJWSEncoder *encoder = [[SecJWSEncoder alloc] init];
    NSMutableData *data = NULL;
    BOOL result = false;

    const uint8_t needsZeroPaddingTo32[] = {
             0x80,0xad,0x23,0x45,0x56,0x12,0xac,0xfe,0xfa,0xce,0x29,0x34,0xfb,0xfa,0x19,
        0x40,0xcc,0xad,0x23,0x45,0x56,0x12,0xac,0xfe,0xfa,0xce,0x29,0x34,0xfb,0xff,0x01
    };
    data = [NSMutableData dataWithCapacity:0];
    result = [encoder appendPaddedToData:data ptr:needsZeroPaddingTo32 len:sizeof(needsZeroPaddingTo32) expected:32];
    XCTAssertEqual(result, YES);
    XCTAssertEqual([data length], 32);

    const uint8_t needsZeroRemovalTo32[] = {
        0x00,0x00,
        0x32,0xcc,0xad,0x23,0x45,0x56,0x12,0xac,0xfe,0xfa,0xce,0x29,0x34,0xfb,0xfa,0x19,
        0x40,0xcc,0xad,0x23,0x45,0x56,0x12,0xac,0xfe,0xfa,0xce,0x29,0x34,0xfb,0xff,0x01
    };
    data = [NSMutableData dataWithCapacity:0];
    result = [encoder appendPaddedToData:data ptr:needsZeroRemovalTo32 len:sizeof(needsZeroRemovalTo32) expected:32];
    XCTAssertEqual(result, YES);
    XCTAssertEqual([data length], 32);

    const uint8_t needsNoPaddingIs32[] = {
        0x00,0x82,0x32,0xcc,0xad,0x23,0x45,0x56,0x12,0xac,0xfe,0xfa,0xce,0x29,0x34,0xfb,
        0x40,0xcc,0xad,0x23,0x45,0x56,0x12,0xac,0xfe,0xfa,0xce,0x29,0x34,0xfb,0xff,0x01
    };
    data = [NSMutableData dataWithCapacity:0];
    result = [encoder appendPaddedToData:data ptr:needsNoPaddingIs32 len:sizeof(needsNoPaddingIs32) expected:32];
    XCTAssertEqual(result, YES);
    XCTAssertEqual([data length], 32);

    const uint8_t cannotBeMadeValid32[] = {
        0x00,0x00,0xaa,0xaa,
        0x00,0x82,0x32,0xcc,0xad,0x23,0x45,0x56,0x12,0xac,0xfe,0xfa,0xce,0x29,0x34,0xfb,
        0x40,0xcc,0xad,0x23,0x45,0x56,0x12,0xac,0xfe,0xfa,0xce,0x29,0x34,0xfb,0xff,0x01
    };
    data = [NSMutableData dataWithCapacity:0];
    result = [encoder appendPaddedToData:data ptr:cannotBeMadeValid32 len:sizeof(cannotBeMadeValid32) expected:32];
    XCTAssertEqual(result, NO); // this case should fail
}

@end
