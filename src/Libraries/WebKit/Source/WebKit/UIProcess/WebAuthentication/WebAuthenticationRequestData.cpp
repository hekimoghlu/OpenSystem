/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 5, 2024.
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
#include "WebAuthenticationRequestData.h"

#if ENABLE(WEB_AUTHN)

#import <WebCore/UserVerificationRequirement.h>

namespace WebKit {
using namespace WebCore;

ClientDataType getClientDataType(const std::variant<PublicKeyCredentialCreationOptions, PublicKeyCredentialRequestOptions>& options)
{
    if (std::holds_alternative<PublicKeyCredentialCreationOptions>(options))
        return ClientDataType::Create;
    return ClientDataType::Get;
}

UserVerificationRequirement getUserVerificationRequirement(const std::variant<PublicKeyCredentialCreationOptions, PublicKeyCredentialRequestOptions>& options)
{
    if (std::holds_alternative<PublicKeyCredentialCreationOptions>(options)) {
        if (auto authenticatorSelection = std::get<PublicKeyCredentialCreationOptions>(options).authenticatorSelection)
            return authenticatorSelection->userVerification;
        return UserVerificationRequirement::Preferred;
    }

    return std::get<PublicKeyCredentialRequestOptions>(options).userVerification;
}

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
