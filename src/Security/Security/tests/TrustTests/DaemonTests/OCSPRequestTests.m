/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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
//  OCSPRequestTests.m
//  Security
//

#include <AssertMacros.h>
#import <XCTest/XCTest.h>
#import <Security/SecCertificatePriv.h>
#include <utilities/SecCFWrappers.h>

#import "trust/trustd/SecOCSPRequest.h"

#import "TrustDaemonTestCase.h"
#import "OCSPRequestTests_data.h"

@interface OCSPRequestTests : TrustDaemonTestCase
@end

@implementation OCSPRequestTests

- (void)testCreate {
    SecCertificateRef leaf = SecCertificateCreateWithBytes(NULL, _ocsp_request_leaf, sizeof(_ocsp_request_leaf));
    SecCertificateRef ca = SecCertificateCreateWithBytes(NULL, _ocsp_request_ca, sizeof(_ocsp_request_ca));

    SecOCSPRequestRef request = SecOCSPRequestCreate(leaf, ca);
    XCTAssert(NULL != request);
    SecOCSPRequestFinalize(request);

    CFReleaseNull(leaf);
    CFReleaseNull(ca);
}

- (void)testGetDER {
    XCTAssert(NULL == SecOCSPRequestGetDER(NULL));

    SecCertificateRef leaf = SecCertificateCreateWithBytes(NULL, _ocsp_request_leaf, sizeof(_ocsp_request_leaf));
    SecCertificateRef ca = SecCertificateCreateWithBytes(NULL, _ocsp_request_ca, sizeof(_ocsp_request_ca));

    SecOCSPRequestRef request = SecOCSPRequestCreate(leaf, ca);
    XCTAssert(NULL != request);

    CFDataRef requestDER = SecOCSPRequestGetDER(request);
    XCTAssert(NULL != requestDER);

    SecOCSPRequestRef decodedRequest = SecOCSPRequestCreateWithDataStrict(requestDER);
    XCTAssert(NULL != decodedRequest);

    //SecOCSPRequestFinalize(decodedRequest);
    SecOCSPRequestFinalize(request);
    CFReleaseNull(leaf);
    CFReleaseNull(ca);
}

@end

