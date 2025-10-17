/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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
#ifndef _TRUSTTESTS_EVALUATION_TESTCASE_H_
#define _TRUSTTESTS_EVALUATION_TESTCASE_H_

#import <XCTest/XCTest.h>
#include <Security/Security.h>
#include "../TrustEvaluationTestHelpers.h"

NS_ASSUME_NONNULL_BEGIN

@interface TrustEvaluationTestCase : XCTestCase
- (id _Nullable)addTrustSettingsForCert:(SecCertificateRef)cert trustSettings:(id)trustSettings; // returns a persistent ref for call to removeTrustSettings, takes a dictionary or array of trust settings
- (id _Nullable)addTrustSettingsForCert:(SecCertificateRef)cert; // returns a persistent ref for call to removeTrustSettings
- (void)removeTrustSettingsForCert:(SecCertificateRef)cert persistentRef:(id _Nullable)persistentRef;
- (void)setTestRootAsSystem:(const uint8_t*)sha256hash; // this is expected to be a 32-byte array
- (void)removeTestRootAsSystem;

// ported from regressionBase
- (void)runCertificateTestForDirectory:(SecPolicyRef)policy subDirectory:(NSString *)resourceSubDirectory verifyDate:(NSDate* _Nullable)date;

- (id _Nullable) CF_RETURNS_RETAINED SecCertificateCreateFromResource:(NSString * )name subdirectory:(NSString *)dir;
- (id _Nullable) CF_RETURNS_RETAINED SecCertificateCreateFromPEMResource:(NSString *)name subdirectory:(NSString *)dir;
@end

/* Use this interface to get a SecCertificateRef that has the same CFTypeID
 * as used by the Security framework */
CF_RETURNS_RETAINED _Nullable
SecCertificateRef SecFrameworkCertificateCreate(const uint8_t * der_bytes, CFIndex der_length);
CF_RETURNS_RETAINED _Nullable
SecCertificateRef SecFrameworkCertificateCreateFromTestCert(SecCertificateRef cert);
CF_RETURNS_RETAINED
SecPolicyRef SecFrameworkPolicyCreateSSL(Boolean server, CFStringRef __nullable hostname);
CF_RETURNS_RETAINED
SecPolicyRef SecFrameworkPolicyCreateBasicX509(void);
CF_RETURNS_RETAINED
SecPolicyRef SecFrameworkPolicyCreateSMIME(CFIndex smimeUsage, CFStringRef __nullable email);
CF_RETURNS_RETAINED
SecPolicyRef SecFrameworkPolicyCreatePassbookCardSigner(CFStringRef cardIssuer, CFStringRef teamIdentifier);

NS_ASSUME_NONNULL_END

#endif /* _TRUSTTESTS_EVALUATION_TESTCASE_H_ */
