/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 3, 2024.
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
#ifndef _TRUSTTEST_TRUST_HELPERS_H_
#define _TRUSTTEST_TRUST_HELPERS_H_

#import <Foundation/Foundation.h>
#import <Security/Security.h>

NS_ASSUME_NONNULL_BEGIN

NSURL * _Nullable setUpTmpDir(void);
int ping_host(char *host_name, char *port);

@interface TestTrustEvaluation : NSObject
@property (assign, nonnull) SecTrustRef trust;
@property NSString *fullTestName;
@property BOOL bridgeOSDisabled;

// Outputs
@property (assign) SecTrustResultType trustResult;
@property (nullable) NSDictionary *resultDictionary;

// Expected results
@property NSNumber *expectedResult;
@property NSNumber *expectedChainLength;

// These properties have the side effect of modifying the SecTrustRef
@property (nullable,assign,nonatomic) NSArray *anchors;
@property (nullable,assign,nonatomic) NSArray *ocspResponses;
@property (nullable,nonatomic) NSArray *presentedSCTs;
@property (nullable,nonatomic) NSArray *trustedCTLogs;
@property (nullable,nonatomic) NSDate *verifyDate;

- (instancetype _Nullable )initWithCertificates:(NSArray * _Nonnull)certs policies:(NSArray * _Nullable)policies;
- (instancetype _Nullable) initWithTrustDictionary:(NSDictionary *)testDict;

- (void)addAnchor:(SecCertificateRef)certificate;
- (void)setNeedsEvaluation;

- (bool)evaluate:(out NSError * _Nullable __autoreleasing * _Nullable)outError;
- (bool)evaluateForExpectedResults:(out NSError * _Nullable __autoreleasing *)outError;
@end

NS_ASSUME_NONNULL_END

#endif /*_TRUSTTEST_TRUST_HELPERS_H_ */
