/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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

#if ENABLE(WEB_AUTHN)
#include "AttestationConveyancePreference.h"
#include "AuthenticationExtensionsClientInputs.h"
#include "BufferSource.h"
#include "PublicKeyCredentialDescriptor.h"
#include "PublicKeyCredentialType.h"
#include "ResidentKeyRequirement.h"
#include "UserVerificationRequirement.h"
#include <wtf/Forward.h>

namespace WebCore {

enum class AuthenticatorAttachment : uint8_t;

struct AuthenticatorSelectionCriteria {
    std::optional<AuthenticatorAttachment> authenticatorAttachment;
    // residentKey replaces requireResidentKey, see: https://www.w3.org/TR/webauthn-2/#dictionary-authenticatorSelection
    std::optional<ResidentKeyRequirement> residentKey;
    bool requireResidentKey { false };
    UserVerificationRequirement userVerification { UserVerificationRequirement::Preferred };
};

}

#endif // ENABLE(WEB_AUTHN)
