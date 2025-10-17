/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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
#import <Foundation/Foundation.h>

#import "RTCMacros.h"

NS_ASSUME_NONNULL_BEGIN

RTC_OBJC_EXPORT
@interface RTCCertificate : NSObject <NSCopying>

/** Private key in PEM. */
@property(nonatomic, readonly, copy) NSString *private_key;

/** Public key in an x509 cert encoded in PEM. */
@property(nonatomic, readonly, copy) NSString *certificate;

/**
 * Initialize an RTCCertificate with PEM strings for private_key and certificate.
 */
- (instancetype)initWithPrivateKey:(NSString *)private_key
                       certificate:(NSString *)certificate NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/** Generate a new certificate for 're' use.
 *
 *  Optional dictionary of parameters. Defaults to KeyType ECDSA if none are
 *  provided.
 *  - name: "ECDSA" or "RSASSA-PKCS1-v1_5"
 */
+ (nullable RTCCertificate *)generateCertificateWithParams:(NSDictionary *)params;

@end

NS_ASSUME_NONNULL_END
