/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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
//  SRPSession.h
//  KeychainCircle
//
//

#import <Foundation/Foundation.h>

#include <corecrypto/ccdigest.h>
#include <corecrypto/ccrng.h>
#include <corecrypto/ccsrp.h>

NS_ASSUME_NONNULL_BEGIN

@interface KCSRPContext : NSObject

- (instancetype) init NS_UNAVAILABLE;

- (instancetype) initWithUser: (NSString*) user
                   digestInfo: (const struct ccdigest_info *) di
                        group: (ccsrp_const_gp_t) gp
                 randomSource: (struct ccrng_state *) rng NS_DESIGNATED_INITIALIZER;

- (bool) isAuthenticated;

// Returns an NSData that refers to the key in the context.
- (NSData* _Nullable) getKey;

@end

@interface KCSRPClientContext : KCSRPContext

- (nullable NSData*) copyStart: (NSError**) error;
- (nullable NSData*) copyResposeToChallenge: (NSData*) B_data
                          password: (NSString*) password
                              salt: (NSData*) salt
                             error: (NSError**) error;
- (bool) verifyConfirmation: (NSData*) HAMK_data
                      error: (NSError**) error;

@end

@interface KCSRPServerContext : KCSRPContext
@property (readonly) NSData* salt;

- (instancetype) initWithUser: (NSString*) user
                         salt: (NSData*) salt
                     verifier: (NSData*) verifier
                   digestInfo: (const struct ccdigest_info *) di
                        group: (ccsrp_const_gp_t) gp
                 randomSource: (struct ccrng_state *) rng NS_DESIGNATED_INITIALIZER;

- (instancetype) initWithUser: (NSString*)user
                     password: (NSString*)password
                   digestInfo: (const struct ccdigest_info *) di
                        group: (ccsrp_const_gp_t) gp
                 randomSource: (struct ccrng_state *) rng NS_DESIGNATED_INITIALIZER;

- (instancetype) initWithUser: (NSString*) user
                   digestInfo: (const struct ccdigest_info *) di
                        group: (ccsrp_const_gp_t) gp
                 randomSource: (struct ccrng_state *) rng NS_UNAVAILABLE;


- (bool) resetWithPassword: (NSString*) password
                     error: (NSError**) error;

- (nullable NSData*) copyChallengeFor: (NSData*) A_data
                       error: (NSError**) error;
- (nullable NSData*) copyConfirmationFor: (NSData*) M_data
                          error: (NSError**) error;

@end

NS_ASSUME_NONNULL_END
