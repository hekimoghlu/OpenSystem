/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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
#pragma once

#ifdef __cplusplus

#import <WebKit/_WKWebAuthenticationPanel.h>

namespace WebCore {
struct PublicKeyCredentialCreationOptions;
struct PublicKeyCredentialRequestOptions;
}

WK_CLASS_AVAILABLE(macos(12.0), ios(15.0))
@interface _WKWebAuthenticationPanel (WKTesting)

+ (WebCore::PublicKeyCredentialCreationOptions)convertToCoreCreationOptionsWithOptions:(_WKPublicKeyCredentialCreationOptions *)options WK_API_AVAILABLE(macos(12.0), ios(15.0));
+ (WebCore::PublicKeyCredentialRequestOptions)convertToCoreRequestOptionsWithOptions:(_WKPublicKeyCredentialRequestOptions *)options WK_API_AVAILABLE(macos(12.0), ios(15.0));

+ (NSArray<NSDictionary *> *)getAllLocalAuthenticatorCredentialsWithAccessGroup:(NSString *)accessGroup WK_API_AVAILABLE(macos(12.0), ios(15.0));

+ (NSArray<NSDictionary *> *)getAllLocalAuthenticatorCredentialsWithRPIDAndAccessGroup:(NSString *)accessGroup rpID:(NSString *)rpID WK_API_AVAILABLE(macos(13.0), ios(16.0));
+ (NSArray<NSDictionary *> *)getAllLocalAuthenticatorCredentialsWithCredentialIDAndAccessGroup:(NSString *)accessGroup credentialID:(NSData *)credentialID WK_API_AVAILABLE(macos(13.0), ios(16.0));

// For details of configuration, refer to MockWebAuthenticationConfiguration.h.
@property (nonatomic, copy) NSDictionary *mockConfiguration;

@end

#endif
