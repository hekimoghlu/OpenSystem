/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 30, 2023.
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

#import "AuthenticationServicesForwardDeclarations.h"
#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_HEADER(WebKit, AuthenticationServices);

SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASAuthorization);
SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASAuthorizationController);
SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASAuthorizationPlatformPublicKeyCredentialProvider);
SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASAuthorizationSecurityKeyPublicKeyCredentialProvider);
SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASAuthorizationWebBrowserPlatformPublicKeyCredentialProvider);
SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASPublicKeyCredentialClientData);
SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASAuthorizationPlatformPublicKeyCredentialRegistration);
SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASAuthorizationSecurityKeyPublicKeyCredentialRegistration);
SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASAuthorizationPlatformPublicKeyCredentialAssertion);
SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASAuthorizationControllerDelegate);
SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASAuthorizationPublicKeyCredentialParameters);
SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASAuthorizationPlatformPublicKeyCredentialDescriptor);
SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASAuthorizationSecurityKeyPublicKeyCredentialDescriptor);
SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASAuthorizationSecurityKeyPublicKeyCredentialAssertion);
SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASAuthorizationPublicKeyCredentialLargeBlobAssertionInput);
SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASAuthorizationPublicKeyCredentialLargeBlobRegistrationInput);
SOFT_LINK_CONSTANT_FOR_HEADER(WebKit, AuthenticationServices, ASAuthorizationErrorDomain, NSErrorDomain);
#if HAVE(WEB_AUTHN_PRF_API)
SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASAuthorizationPublicKeyCredentialPRFRegistrationInput);
SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASAuthorizationPublicKeyCredentialPRFAssertionInputValues);
SOFT_LINK_CLASS_FOR_HEADER(WebKit, ASAuthorizationPublicKeyCredentialPRFAssertionInput);
#endif
#define ASAuthorizationErrorDomain WebKit::get_AuthenticationServices_ASAuthorizationErrorDomain()
