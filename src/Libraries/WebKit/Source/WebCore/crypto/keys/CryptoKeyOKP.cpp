/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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
#include "CryptoKeyOKP.h"

#include "CryptoAlgorithmRegistry.h"
#include "JsonWebKey.h"
#include <wtf/text/Base64.h>

namespace WebCore {

static const ASCIILiteral X25519 { "X25519"_s };
static const ASCIILiteral Ed25519 { "Ed25519"_s };

static constexpr size_t keySizeInBytesFromNamedCurve(CryptoKeyOKP::NamedCurve curve)
{
    switch (curve) {
    case CryptoKeyOKP::NamedCurve::X25519:
    case CryptoKeyOKP::NamedCurve::Ed25519:
        return 32;
    }
    return 32;
}

RefPtr<CryptoKeyOKP> CryptoKeyOKP::create(CryptoAlgorithmIdentifier identifier, NamedCurve curve, CryptoKeyType type, KeyMaterial&& platformKey, bool extractable, CryptoKeyUsageBitmap usages)
{
    if (platformKey.size() != keySizeInBytesFromNamedCurve(curve))
        return nullptr;
    return adoptRef(*new CryptoKeyOKP(identifier, curve, type, WTFMove(platformKey), extractable, usages));
}

CryptoKeyOKP::CryptoKeyOKP(CryptoAlgorithmIdentifier identifier, NamedCurve curve, CryptoKeyType type, KeyMaterial&& data, bool extractable, CryptoKeyUsageBitmap usages)
    : CryptoKey(identifier, type, extractable, usages)
    , m_curve(curve)
    , m_data(WTFMove(data))
{
}

ExceptionOr<CryptoKeyPair> CryptoKeyOKP::generatePair(CryptoAlgorithmIdentifier identifier, NamedCurve namedCurve, bool extractable, CryptoKeyUsageBitmap usages)
{
    if (!supportsNamedCurve())
        return Exception { ExceptionCode::NotSupportedError };

    auto result = platformGeneratePair(identifier, namedCurve, extractable, usages);
    if (!result)
        return Exception { ExceptionCode::OperationError };

    return WTFMove(*result);
}

RefPtr<CryptoKeyOKP> CryptoKeyOKP::importRaw(CryptoAlgorithmIdentifier identifier, NamedCurve namedCurve, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap usages)
{
    if (!supportsNamedCurve())
        return nullptr;

    // FIXME: The Ed25519 spec states that import in raw format must be used only for Verify.
    return create(identifier, namedCurve, usages & CryptoKeyUsageSign ? CryptoKeyType::Private : CryptoKeyType::Public, WTFMove(keyData), extractable, usages);
}

RefPtr<CryptoKeyOKP> CryptoKeyOKP::importJwk(CryptoAlgorithmIdentifier identifier, NamedCurve namedCurve, JsonWebKey&& keyData, bool extractable, CryptoKeyUsageBitmap usages)
{
    if (!supportsNamedCurve())
        return nullptr;

    switch (namedCurve) {
    case NamedCurve::Ed25519:
        // FIXME: this is already done in the Algorithm's importKey method for each format, so it seems we can remoev this duplicated code.
        if (!keyData.d.isEmpty()) {
            if (usages & (CryptoKeyUsageEncrypt | CryptoKeyUsageDecrypt | CryptoKeyUsageVerify | CryptoKeyUsageDeriveKey | CryptoKeyUsageDeriveBits | CryptoKeyUsageWrapKey | CryptoKeyUsageUnwrapKey))
                return nullptr;
        } else {
            if (usages & (CryptoKeyUsageEncrypt | CryptoKeyUsageDecrypt | CryptoKeyUsageSign | CryptoKeyUsageDeriveKey | CryptoKeyUsageDeriveBits | CryptoKeyUsageWrapKey | CryptoKeyUsageUnwrapKey))
                return nullptr;
        }
        if (keyData.crv != "Ed25519"_s)
            return nullptr;
        // FIXME: Do we have tests for these checks ?
        if (!keyData.alg.isEmpty() && keyData.alg != "EdDSA"_s)
            return nullptr;
        if (usages && !keyData.use.isEmpty() && keyData.use != "sign"_s)
            return nullptr;
        if (keyData.key_ops && ((keyData.usages & usages) != usages))
            return nullptr;
        if (keyData.ext && !keyData.ext.value() && extractable)
            return nullptr;
        break;
    case NamedCurve::X25519:
        if (keyData.crv != "X25519"_s)
            return nullptr;
        if (keyData.key_ops && ((keyData.usages & usages) != usages))
            return nullptr;
        if (keyData.ext && !keyData.ext.value() && extractable)
            return nullptr;
        break;
    }

    if (keyData.kty != "OKP"_s)
        return nullptr;

    if (keyData.x.isNull())
        return nullptr;

    auto x = base64URLDecode(keyData.x);
    if (!x)
        return nullptr;

    if (!keyData.d.isNull()) {
        auto d = base64URLDecode(keyData.d);
        if (!d || !platformCheckPairedKeys(identifier, namedCurve, *d, *x))
            return nullptr;
        return create(identifier, namedCurve, CryptoKeyType::Private, WTFMove(*d), extractable, usages);
    }

    return create(identifier, namedCurve, CryptoKeyType::Public, WTFMove(*x), extractable, usages);
}

ExceptionOr<Vector<uint8_t>> CryptoKeyOKP::exportRaw() const
{
    if (type() != CryptoKey::Type::Public)
        return Exception { ExceptionCode::InvalidAccessError };

    auto result = platformExportRaw();
    if (result.isEmpty())
        return Exception { ExceptionCode::OperationError };
    return result;
}

ExceptionOr<JsonWebKey> CryptoKeyOKP::exportJwk() const
{
    JsonWebKey result;
    result.kty = "OKP"_s;
    switch (m_curve) {
    case NamedCurve::X25519:
        result.crv = X25519;
        break;
    case NamedCurve::Ed25519:
        result.crv = Ed25519;
        break;
    }

    result.key_ops = usages();
    result.usages = usagesBitmap();
    result.ext = extractable();

    switch (type()) {
    case CryptoKeyType::Private:
        result.d = generateJwkD();
        result.x = generateJwkX();
        break;
    case CryptoKeyType::Public:
        result.x = generateJwkX();
        break;
    case CryptoKeyType::Secret:
        return Exception { ExceptionCode::OperationError };
    }

    return result;
}

std::optional<CryptoKeyOKP::NamedCurve> CryptoKeyOKP::namedCurveFromString(const String& curveString)
{
    if (curveString == X25519)
        return NamedCurve::X25519;

    if (curveString == Ed25519)
        return NamedCurve::Ed25519;

    return std::nullopt;
}

String CryptoKeyOKP::namedCurveString() const
{
    switch (m_curve) {
    case NamedCurve::X25519:
        return X25519;
    case NamedCurve::Ed25519:
        return Ed25519;
    }

    ASSERT_NOT_REACHED();
    return emptyString();
}

bool CryptoKeyOKP::isValidOKPAlgorithm(CryptoAlgorithmIdentifier algorithm)
{
    return algorithm == CryptoAlgorithmIdentifier::Ed25519;
}

auto CryptoKeyOKP::algorithm() const -> KeyAlgorithm
{
    return CryptoKeyAlgorithm { CryptoAlgorithmRegistry::singleton().name(algorithmIdentifier()) };
}

CryptoKey::Data CryptoKeyOKP::data() const
{
    auto key = platformKey();
    return CryptoKey::Data {
        CryptoKeyClass::OKP,
        algorithmIdentifier(),
        extractable(),
        usagesBitmap(),
        WTFMove(key),
        std::nullopt,
        std::nullopt,
        namedCurveString(),
        std::nullopt,
        type()
    };
}

#if !PLATFORM(COCOA) && !USE(GCRYPT)

bool CryptoKeyOKP::supportsNamedCurve()
{
    return false;
}

std::optional<CryptoKeyPair> CryptoKeyOKP::platformGeneratePair(CryptoAlgorithmIdentifier, NamedCurve, bool, CryptoKeyUsageBitmap)
{
    return { };
}

bool CryptoKeyOKP::platformCheckPairedKeys(CryptoAlgorithmIdentifier, NamedCurve, const Vector<uint8_t>&, const Vector<uint8_t>&)
{
    return true;
}

RefPtr<CryptoKeyOKP> CryptoKeyOKP::importSpki(CryptoAlgorithmIdentifier, NamedCurve, Vector<uint8_t>&&, bool, CryptoKeyUsageBitmap)
{
    // FIXME: Implement it.
    return nullptr;
}

ExceptionOr<Vector<uint8_t>> CryptoKeyOKP::exportSpki() const
{
    return Exception { ExceptionCode::NotSupportedError };
}

RefPtr<CryptoKeyOKP> CryptoKeyOKP::importPkcs8(CryptoAlgorithmIdentifier, NamedCurve, Vector<uint8_t>&&, bool, CryptoKeyUsageBitmap)
{
    // FIXME: Implement it.
    return nullptr;
}

ExceptionOr<Vector<uint8_t>> CryptoKeyOKP::exportPkcs8() const
{
    return Exception { ExceptionCode::NotSupportedError };
}

String CryptoKeyOKP::generateJwkD() const
{
    return { };
}

String CryptoKeyOKP::generateJwkX() const
{
    return { };
}

Vector<uint8_t> CryptoKeyOKP::platformExportRaw() const
{
    return { };
}
#endif

} // namespace WebCore
