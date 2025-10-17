/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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
#include "PushCrypto.h"

#include <wtf/Scope.h>

namespace WebCore::PushCrypto {

#if !PLATFORM(COCOA)

P256DHKeyPair P256DHKeyPair::generate(void)
{
    return { };
}

bool validateP256DHPublicKey(std::span<const uint8_t>)
{
    return false;
}

std::optional<Vector<uint8_t>> computeP256DHSharedSecret(std::span<const uint8_t>, const P256DHKeyPair&)
{
    return std::nullopt;
}

Vector<uint8_t> hmacSHA256(std::span<const uint8_t>, std::span<const uint8_t>)
{
    return { };
}

std::optional<Vector<uint8_t>> decryptAES128GCM(std::span<const uint8_t>, std::span<const uint8_t>, std::span<const uint8_t>)
{
    return std::nullopt;
}

#endif // !PLATFORM(COCOA)

} // namespace WebCore::PushCrypto
