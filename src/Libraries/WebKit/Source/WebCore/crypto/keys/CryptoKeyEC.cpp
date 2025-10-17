/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 19, 2023.
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
#include "CryptoKeyEC.h"

#include "CryptoAlgorithmRegistry.h"
#include "JsonWebKey.h"
#if HAVE(SWIFT_CPP_INTEROP)
#include <pal/PALSwift.h>
#endif
#include <wtf/text/Base64.h>

namespace WebCore {

static const ASCIILiteral P256 { "P-256"_s };
static const ASCIILiteral P384 { "P-384"_s };
static const ASCIILiteral P521 { "P-521"_s };

static std::optional<CryptoKeyEC::NamedCurve> toNamedCurve(const String& curve)
{
    if (curve == P256)
        return CryptoKeyEC::NamedCurve::P256;
    if (curve == P384)
        return CryptoKeyEC::NamedCurve::P384;
    if (curve == P521)
        return CryptoKeyEC::NamedCurve::P521;

    return std::nullopt;
}

CryptoKeyEC::CryptoKeyEC(CryptoAlgorithmIdentifier identifier, NamedCurve curve, CryptoKeyType type, PlatformECKeyContainer&& platformKey, bool extractable, CryptoKeyUsageBitmap usages)
    : CryptoKey(identifier, type, extractable, usages)
    , m_platformKey(WTFMove(platformKey))
    , m_curve(curve)
{
    // Only CryptoKeyEC objects for supported curves should be created.
    ASSERT(platformSupportedCurve(curve));
}

CryptoKeyEC::~CryptoKeyEC() = default;

ExceptionOr<CryptoKeyPair> CryptoKeyEC::generatePair(CryptoAlgorithmIdentifier identifier, const String& curve, bool extractable, CryptoKeyUsageBitmap usages)
{
    auto namedCurve = toNamedCurve(curve);
    if (!namedCurve || !platformSupportedCurve(*namedCurve))
        return Exception { ExceptionCode::NotSupportedError };

    auto result = platformGeneratePair(identifier, *namedCurve, extractable, usages);
    if (!result)
        return Exception { ExceptionCode::OperationError };

    return WTFMove(*result);
}

RefPtr<CryptoKeyEC> CryptoKeyEC::importRaw(CryptoAlgorithmIdentifier identifier, const String& curve, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap usages)
{
    auto namedCurve = toNamedCurve(curve);
    if (!namedCurve || !platformSupportedCurve(*namedCurve))
        return nullptr;

    return platformImportRaw(identifier, *namedCurve, WTFMove(keyData), extractable, usages);
}

RefPtr<CryptoKeyEC> CryptoKeyEC::importJwk(CryptoAlgorithmIdentifier identifier, const String& curve, JsonWebKey&& keyData, bool extractable, CryptoKeyUsageBitmap usages)
{
    if (keyData.kty != "EC"_s)
        return nullptr;
    if (keyData.key_ops && ((keyData.usages & usages) != usages))
        return nullptr;
    if (keyData.ext && !keyData.ext.value() && extractable)
        return nullptr;

    if (keyData.crv.isNull() || curve != keyData.crv)
        return nullptr;
    auto namedCurve = toNamedCurve(keyData.crv);
    if (!namedCurve || !platformSupportedCurve(*namedCurve))
        return nullptr;

    if (keyData.x.isNull() || keyData.y.isNull())
        return nullptr;
    auto x = base64URLDecode(keyData.x);
    if (!x)
        return nullptr;
    auto y = base64URLDecode(keyData.y);
    if (!y)
        return nullptr;
    if (keyData.d.isNull()) {
        // import public key
        return platformImportJWKPublic(identifier, *namedCurve, WTFMove(*x), WTFMove(*y), extractable, usages);
    }

    auto d = base64URLDecode(keyData.d);
    if (!d)
        return nullptr;
    // import private key
    return platformImportJWKPrivate(identifier, *namedCurve, WTFMove(*x), WTFMove(*y), WTFMove(*d), extractable, usages);
}

RefPtr<CryptoKeyEC> CryptoKeyEC::importSpki(CryptoAlgorithmIdentifier identifier, const String& curve, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap usages)
{
    auto namedCurve = toNamedCurve(curve);
    if (!namedCurve || !platformSupportedCurve(*namedCurve))
        return nullptr;

    return platformImportSpki(identifier, *namedCurve, WTFMove(keyData), extractable, usages);
}

RefPtr<CryptoKeyEC> CryptoKeyEC::importPkcs8(CryptoAlgorithmIdentifier identifier, const String& curve, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap usages)
{
    auto namedCurve = toNamedCurve(curve);
    if (!namedCurve || !platformSupportedCurve(*namedCurve))
        return nullptr;

    return platformImportPkcs8(identifier, *namedCurve, WTFMove(keyData), extractable, usages);
}

ExceptionOr<Vector<uint8_t>> CryptoKeyEC::exportRaw() const
{
    if (type() != CryptoKey::Type::Public)
        return Exception { ExceptionCode::InvalidAccessError };

    auto&& result = platformExportRaw();
    if (result.isEmpty())
        return Exception { ExceptionCode::OperationError };
    return WTFMove(result);
}

ExceptionOr<JsonWebKey> CryptoKeyEC::exportJwk() const
{
    JsonWebKey result;
    result.kty = "EC"_s;
    switch (m_curve) {
    case NamedCurve::P256:
        result.crv = P256;
        break;
    case NamedCurve::P384:
        result.crv = P384;
        break;
    case NamedCurve::P521:
        result.crv = P521;
        break;
    }
    result.key_ops = usages();
    result.usages = usagesBitmap();
    result.ext = extractable();
    if (!platformAddFieldElements(result))
        return Exception { ExceptionCode::OperationError };
    return result;
}

ExceptionOr<Vector<uint8_t>> CryptoKeyEC::exportSpki() const
{
    if (type() != CryptoKey::Type::Public)
        return Exception { ExceptionCode::InvalidAccessError };

    auto&& result = platformExportSpki();
    if (result.isEmpty())
        return Exception { ExceptionCode::OperationError };
    return WTFMove(result);
}

ExceptionOr<Vector<uint8_t>> CryptoKeyEC::exportPkcs8() const
{
    if (type() != CryptoKey::Type::Private)
        return Exception { ExceptionCode::InvalidAccessError };

    auto&& result = platformExportPkcs8();
    if (result.isEmpty())
        return Exception { ExceptionCode::OperationError };
    return WTFMove(result);
}

String CryptoKeyEC::namedCurveString() const
{
    switch (m_curve) {
    case NamedCurve::P256:
        return String(P256);
    case NamedCurve::P384:
        return String(P384);
    case NamedCurve::P521:
        return String(P521);
    }

    ASSERT_NOT_REACHED();
    return emptyString();
}

bool CryptoKeyEC::isValidECAlgorithm(CryptoAlgorithmIdentifier algorithm)
{
    return algorithm == CryptoAlgorithmIdentifier::ECDSA || algorithm == CryptoAlgorithmIdentifier::ECDH;
}

auto CryptoKeyEC::algorithm() const -> KeyAlgorithm
{
    CryptoEcKeyAlgorithm result;
    result.name = CryptoAlgorithmRegistry::singleton().name(algorithmIdentifier());

    switch (m_curve) {
    case NamedCurve::P256:
        result.namedCurve = P256;
        break;
    case NamedCurve::P384:
        result.namedCurve = P384;
        break;
    case NamedCurve::P521:
        result.namedCurve = P521;
        break;
    }

    return result;
}

CryptoKey::Data CryptoKeyEC::data() const
{
    auto jwkOrException = exportJwk();
    auto jwk = jwkOrException.hasException() ? std::nullopt : std::optional<JsonWebKey> { jwkOrException.releaseReturnValue() };
    return CryptoKey::Data {
        CryptoKeyClass::EC,
        algorithmIdentifier(),
        extractable(),
        usagesBitmap(),
        std::nullopt,
        WTFMove(jwk),
        std::nullopt,
        namedCurveString(),
        std::nullopt,
        type()
    };
}


} // namespace WebCore
