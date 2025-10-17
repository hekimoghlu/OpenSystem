/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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

#import <WebKit/WKFoundation.h>

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class _WKAuthenticationExtensionsClientInputs;
@class _WKAuthenticatorSelectionCriteria;
@class _WKPublicKeyCredentialDescriptor;
@class _WKPublicKeyCredentialParameters;
@class _WKPublicKeyCredentialRelyingPartyEntity;
@class _WKPublicKeyCredentialUserEntity;

typedef NS_ENUM(NSInteger, _WKAttestationConveyancePreference) {
    _WKAttestationConveyancePreferenceNone,
    _WKAttestationConveyancePreferenceIndirect,
    _WKAttestationConveyancePreferenceDirect,
    _WKAttestationConveyancePreferenceEnterprise,
} WK_API_AVAILABLE(macos(12.0), ios(15.0));

WK_CLASS_AVAILABLE(macos(12.0), ios(15.0))
@interface _WKPublicKeyCredentialCreationOptions : NSObject

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithRelyingParty:(_WKPublicKeyCredentialRelyingPartyEntity *)relyingParty user:(_WKPublicKeyCredentialUserEntity *)user publicKeyCredentialParamaters:(NSArray<_WKPublicKeyCredentialParameters *> *)publicKeyCredentialParamaters;

@property (nonatomic, strong) _WKPublicKeyCredentialRelyingPartyEntity *relyingParty;
@property (nonatomic, strong) _WKPublicKeyCredentialUserEntity *user;

@property (nonatomic, copy) NSArray<_WKPublicKeyCredentialParameters *> *publicKeyCredentialParamaters;

@property (nullable, nonatomic, copy) NSNumber *timeout;
@property (nullable, nonatomic, copy) NSArray<_WKPublicKeyCredentialDescriptor *> *excludeCredentials;
@property (nullable, nonatomic, strong) _WKAuthenticatorSelectionCriteria *authenticatorSelection;

/*!@discussion The default value is _WKAttestationConveyancePrefenprenceNone.*/
@property (nonatomic) _WKAttestationConveyancePreference attestation;
@property (nullable, nonatomic, strong) _WKAuthenticationExtensionsClientInputs *extensions;
@property (nullable, nonatomic, strong) NSData *extensionsCBOR;

@end

NS_ASSUME_NONNULL_END
