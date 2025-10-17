/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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
#include "CryptoKey.h"

#include "CryptoAlgorithmRegistry.h"
#include "CryptoKeyAES.h"
#include "CryptoKeyEC.h"
#include "CryptoKeyHMAC.h"
#include "CryptoKeyOKP.h"
#include "CryptoKeyRSA.h"
#include "CryptoKeyRaw.h"
#include "WebCoreOpaqueRoot.h"
#include <wtf/CryptographicallyRandomNumber.h>

namespace WebCore {

CryptoKey::CryptoKey(CryptoAlgorithmIdentifier algorithmIdentifier, Type type, bool extractable, CryptoKeyUsageBitmap usages)
    : m_algorithmIdentifier(algorithmIdentifier)
    , m_type(type)
    , m_extractable(extractable)
    , m_usages(usages)
{
}

CryptoKey::~CryptoKey() = default;

auto CryptoKey::usages() const -> Vector<CryptoKeyUsage>
{
    // The result is ordered alphabetically.
    Vector<CryptoKeyUsage> result;
    if (m_usages & CryptoKeyUsageDecrypt)
        result.append(CryptoKeyUsage::Decrypt);
    if (m_usages & CryptoKeyUsageDeriveBits)
        result.append(CryptoKeyUsage::DeriveBits);
    if (m_usages & CryptoKeyUsageDeriveKey)
        result.append(CryptoKeyUsage::DeriveKey);
    if (m_usages & CryptoKeyUsageEncrypt)
        result.append(CryptoKeyUsage::Encrypt);
    if (m_usages & CryptoKeyUsageSign)
        result.append(CryptoKeyUsage::Sign);
    if (m_usages & CryptoKeyUsageUnwrapKey)
        result.append(CryptoKeyUsage::UnwrapKey);
    if (m_usages & CryptoKeyUsageVerify)
        result.append(CryptoKeyUsage::Verify);
    if (m_usages & CryptoKeyUsageWrapKey)
        result.append(CryptoKeyUsage::WrapKey);
    return result;
}

WebCoreOpaqueRoot root(CryptoKey* key)
{
    return WebCoreOpaqueRoot { key };
}

#if !OS(DARWIN) || PLATFORM(GTK)
Vector<uint8_t> CryptoKey::randomData(size_t size)
{
    Vector<uint8_t> result(size);
    cryptographicallyRandomValues(result.mutableSpan());
    return result;
}
#endif

RefPtr<CryptoKey> CryptoKey::create(CryptoKey::Data&& data)
{
    switch (data.keyClass) {
    case CryptoKeyClass::AES: {
        if (data.jwk)
            return CryptoKeyAES::importJwk(data.algorithmIdentifier, WTFMove(*data.jwk), data.extractable, data.usages, [](auto, auto) { return true; });
        break;
    }
    case CryptoKeyClass::EC: {
        if (data.namedCurveString && data.jwk)
            return CryptoKeyEC::importJwk(data.algorithmIdentifier, *data.namedCurveString, WTFMove(*data.jwk), data.extractable, data.usages);
        break;
    }
    case CryptoKeyClass::HMAC:
        if (data.hashAlgorithmIdentifier && data.lengthBits && data.jwk)
            return CryptoKeyHMAC::importJwk(*data.lengthBits, *data.hashAlgorithmIdentifier, WTFMove(*data.jwk), data.extractable, data.usages, [](auto, auto) { return true; });
        break;
    case CryptoKeyClass::OKP:
        if (data.namedCurveString && data.key && data.type) {
            if (auto namedCurve = CryptoKeyOKP::namedCurveFromString(*data.namedCurveString))
                return CryptoKeyOKP::create(data.algorithmIdentifier, *namedCurve, *data.type, WTFMove(*data.key), data.extractable, data.usages);
        }
        break;
    case CryptoKeyClass::RSA: {
        if (data.jwk)
            return CryptoKeyRSA::importJwk(data.algorithmIdentifier, data.hashAlgorithmIdentifier, WTFMove(*data.jwk), data.extractable, data.usages);
        break;
    }
    case CryptoKeyClass::Raw:
        if (data.key)
            return CryptoKeyRaw::create(data.algorithmIdentifier, WTFMove(*data.key), data.usages);
        break;
    }

    return nullptr;
}

} // namespace WebCore
