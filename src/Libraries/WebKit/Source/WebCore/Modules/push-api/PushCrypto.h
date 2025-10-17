/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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

#include <span>
#include <wtf/Forward.h>
#include <wtf/Vector.h>

namespace WebCore::PushCrypto {

static constexpr size_t p256dhPublicKeyLength = 65;
static constexpr size_t p256dhPrivateKeyLength = 32;
static constexpr size_t p256dhSharedSecretLength = 32;
static constexpr size_t sha256DigestLength = 32;
static constexpr size_t aes128GCMTagLength = 16;

struct P256DHKeyPair {
    Vector<uint8_t> publicKey;
    Vector<uint8_t> privateKey;

    static P256DHKeyPair generate(void);
};

bool validateP256DHPublicKey(std::span<const uint8_t> publicKey);

WEBCORE_EXPORT std::optional<Vector<uint8_t>> computeP256DHSharedSecret(std::span<const uint8_t> publicKey, const P256DHKeyPair&);

WEBCORE_EXPORT Vector<uint8_t> hmacSHA256(std::span<const uint8_t> key, std::span<const uint8_t> message);

WEBCORE_EXPORT std::optional<Vector<uint8_t>> decryptAES128GCM(std::span<const uint8_t> key, std::span<const uint8_t> iv, std::span<const uint8_t> cipherTextWithTag);

} // namespace WebCore::PushCrypto
