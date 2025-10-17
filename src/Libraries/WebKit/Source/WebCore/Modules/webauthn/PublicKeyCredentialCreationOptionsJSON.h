/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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
#include "AuthenticationExtensionsClientInputsJSON.h"
#include "PublicKeyCredentialRpEntity.h"
#include "PublicKeyCredentialUserEntityJSON.h"
#include <wtf/Forward.h>

namespace WebCore {

enum class AuthenticatorAttachment : uint8_t;
enum class AttestationConveyancePreference : uint8_t;
struct AuthenticatorSelectionCriteria;
struct PublicKeyCredentialDescriptorJSON;
struct PublicKeyCredentialParameters;

struct PublicKeyCredentialCreationOptionsJSON {
    PublicKeyCredentialRpEntity rp;
    PublicKeyCredentialUserEntityJSON user;

    String challenge;
    mutable Vector<PublicKeyCredentialParameters> pubKeyCredParams;

    std::optional<unsigned> timeout;
    Vector<PublicKeyCredentialDescriptorJSON> excludeCredentials;
    std::optional<AuthenticatorSelectionCriteria> authenticatorSelection;
    String attestation;
    std::optional<AuthenticationExtensionsClientInputsJSON> extensions;
};

} // namespace WebCore
#endif // ENABLE(WEB_AUTHN)
