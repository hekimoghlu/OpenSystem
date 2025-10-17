/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 27, 2023.
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

#include "PushCrypto.h"
#include <span>
#include <wtf/Forward.h>
#include <wtf/Vector.h>

namespace WebCore::PushCrypto {

struct ClientKeys {
    P256DHKeyPair clientP256DHKeyPair;
    Vector<uint8_t> sharedAuthSecret;

    WEBCORE_EXPORT static ClientKeys generate();
};

// Decrypts a push payload encoded with the aes128gcm Content-Encoding as described in RFC8291.
WEBCORE_EXPORT std::optional<Vector<uint8_t>> decryptAES128GCMPayload(const ClientKeys&, std::span<const uint8_t> payload);

// Decrypts a push payload encoded with the aesgcm Content-Encoding as described in draft-ietf-webpush-encryption-04.
WEBCORE_EXPORT std::optional<Vector<uint8_t>> decryptAESGCMPayload(const ClientKeys&, std::span<const uint8_t> serverP256DHPublicKey, std::span<const uint8_t> salt, std::span<const uint8_t> payload);

} // namespace WebCore::PushCrypto
