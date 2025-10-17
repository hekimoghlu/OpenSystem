/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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
#include "CryptoAlgorithmAESKW.h"

#include "CryptoKeyAES.h"
#include "OpenSSLUtilities.h"

namespace WebCore {

static std::optional<Vector<uint8_t>> cryptWrapKey(const Vector<uint8_t>& key, const Vector<uint8_t>& data)
{
    if (data.size() % 8)
        return std::nullopt;

    AESKey aesKey;
    if (!aesKey.setKey(key, AES_ENCRYPT))
        return std::nullopt;

    Vector<uint8_t> cipherText(data.size() + 8);
    if (AES_wrap_key(aesKey.key(), nullptr, cipherText.data(), data.data(), data.size()) < 0)
        return std::nullopt;

    return cipherText;
}

static std::optional<Vector<uint8_t>> cryptUnwrapKey(const Vector<uint8_t>& key, const Vector<uint8_t>& data)
{
    if (data.size() % 8 || !data.size())
        return std::nullopt;

    AESKey aesKey;
    if (!aesKey.setKey(key, AES_DECRYPT))
        return std::nullopt;

    Vector<uint8_t> plainText(data.size() - 8);
    if (AES_unwrap_key(aesKey.key(), nullptr, plainText.data(), data.data(), data.size()) < 0)
        return std::nullopt;

    return plainText;
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESKW::platformWrapKey(const CryptoKeyAES& key, const Vector<uint8_t>& data)
{
    auto output = cryptWrapKey(key.key(), data);
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*output);
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESKW::platformUnwrapKey(const CryptoKeyAES& key, const Vector<uint8_t>& data)
{
    auto output = cryptUnwrapKey(key.key(), data);
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*output);
}

} // namespace WebCore
