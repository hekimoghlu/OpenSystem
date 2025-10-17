/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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
#include "config.h"

#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_SOURCE(WebKit, AuthenticationServices);

SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServices, ASAuthorizationController);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServices, ASAuthorizationPlatformPublicKeyCredentialProvider);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServices, ASAuthorizationSecurityKeyPublicKeyCredentialProvider);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServices, ASPublicKeyCredentialClientData);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServices, ASAuthorizationPlatformPublicKeyCredentialRegistration);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServices, ASAuthorizationSecurityKeyPublicKeyCredentialRegistration);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServices, ASAuthorizationPlatformPublicKeyCredentialAssertion);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServices, ASAuthorizationPublicKeyCredentialParameters);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServices, ASAuthorizationPlatformPublicKeyCredentialDescriptor);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServices, ASAuthorizationSecurityKeyPublicKeyCredentialDescriptor);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServices, ASAuthorizationSecurityKeyPublicKeyCredentialAssertion);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServices, ASAuthorizationPublicKeyCredentialLargeBlobAssertionInput);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServices, ASAuthorizationPublicKeyCredentialLargeBlobRegistrationInput);
SOFT_LINK_CONSTANT_FOR_SOURCE(WebKit, AuthenticationServices, ASAuthorizationErrorDomain, NSErrorDomain);
#if HAVE(WEB_AUTHN_PRF_API)
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServices, ASAuthorizationPublicKeyCredentialPRFRegistrationInput);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServices, ASAuthorizationPublicKeyCredentialPRFAssertionInputValues);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServices, ASAuthorizationPublicKeyCredentialPRFAssertionInput);
#endif
