/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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
#import <WebKit/_WKAuthenticatorAttachment.h>
#import <WebKit/_WKUserVerificationRequirement.h>

NS_ASSUME_NONNULL_BEGIN

@class _WKAuthenticationExtensionsClientInputs;
@class _WKPublicKeyCredentialDescriptor;

WK_CLASS_AVAILABLE(macos(12.0), ios(15.0))
@interface _WKPublicKeyCredentialRequestOptions : NSObject

@property (nullable, nonatomic, copy) NSNumber *timeout;
@property (nullable, nonatomic, copy) NSString *relyingPartyIdentifier;
@property (nullable, nonatomic, copy) NSArray<_WKPublicKeyCredentialDescriptor *> *allowCredentials;
/*!@discussion The default value is _WKUserVerificationRequirementPreferred.*/
@property (nonatomic) _WKUserVerificationRequirement userVerification;
/*!@discussion The default value is _WKAuthenticatorAttachmentAll.*/
@property (nonatomic) _WKAuthenticatorAttachment authenticatorAttachment;
@property (nullable, nonatomic, strong) _WKAuthenticationExtensionsClientInputs *extensions;
@property (nullable, nonatomic, copy) NSData *extensionsCBOR;

@end

NS_ASSUME_NONNULL_END
