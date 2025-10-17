/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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

#include "AuthenticationExtensionsClientOutputsJSON.h"
#include "IDLTypes.h"
#include <wtf/Forward.h>

namespace WebCore {

struct RegistrationResponseJSON {
    struct AuthenticatorAttestationResponseJSON {
        String clientDataJSON;
        String authenticatorData;
        Vector<String> transports;
        String publicKey;
        long long publicKeyAlgorithm { 0 };
        String attestationObject;
    };
    String id;
    String rawId;
    AuthenticatorAttestationResponseJSON response;
    String authenticatorAttachment;
    AuthenticationExtensionsClientOutputsJSON clientExtensionResults;
    String type;
};
}

#endif // ENABLE(WEB_AUTHN)
