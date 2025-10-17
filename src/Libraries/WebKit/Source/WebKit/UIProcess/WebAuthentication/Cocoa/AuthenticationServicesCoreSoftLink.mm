/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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

#import "AuthenticationServicesCoreSPI.h"
#import <wtf/SoftLinking.h>

SOFT_LINK_PRIVATE_FRAMEWORK_FOR_SOURCE(WebKit, AuthenticationServicesCore);

SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, AuthenticationServicesCore, ASCWebKitSPISupport);

SOFT_LINK_CONSTANT_FOR_SOURCE(WebKit, AuthenticationServicesCore, ASCAuthorizationErrorDomain, NSErrorDomain);

#if HAVE(ASC_AUTH_UI) || HAVE(UNIFIED_ASC_AUTH_UI)

SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServicesCore, ASCAgentProxy);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServicesCore, ASCAppleIDCredential);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServicesCore, ASCAuthorizationPresentationContext);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServicesCore, ASCAuthorizationPresenter);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServicesCore, ASCAuthorizationRemotePresenter);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServicesCore, ASCCredentialRequestContext);
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, AuthenticationServicesCore, ASCWebAuthenticationExtensionsClientInputs);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServicesCore, ASCPlatformPublicKeyCredentialAssertion);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServicesCore, ASCPlatformPublicKeyCredentialLoginChoice);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServicesCore, ASCPlatformPublicKeyCredentialRegistration);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServicesCore, ASCPublicKeyCredentialAssertionOptions);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServicesCore, ASCPublicKeyCredentialCreationOptions);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServicesCore, ASCPublicKeyCredentialDescriptor);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServicesCore, ASCSecurityKeyPublicKeyCredentialAssertion);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServicesCore, ASCSecurityKeyPublicKeyCredentialLoginChoice);
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, AuthenticationServicesCore, ASCSecurityKeyPublicKeyCredentialRegistration);
SOFT_LINK_CLASS_FOR_SOURCE_OPTIONAL(WebKit, AuthenticationServicesCore, ASGlobalFrameIdentifier);

SOFT_LINK_CONSTANT_FOR_SOURCE(WebKit, AuthenticationServicesCore, ASCPINValidationResultKey, NSString*);

#endif // HAVE(ASC_AUTH_UI) || HAVE(UNIFIED_ASC_AUTH_UI)
