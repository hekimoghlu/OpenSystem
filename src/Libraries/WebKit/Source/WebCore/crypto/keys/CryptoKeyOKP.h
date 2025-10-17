/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 29, 2023.
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

#include "CryptoKey.h"
#include "CryptoKeyPair.h"
#include "ExceptionOr.h"

namespace WebCore {

struct JsonWebKey;

class CryptoKeyOKP final : public CryptoKey {
public:
    using KeyMaterial = Vector<uint8_t>;

    enum class NamedCurve : bool {
        X25519,
        Ed25519,
    };

    static RefPtr<CryptoKeyOKP> create(CryptoAlgorithmIdentifier, NamedCurve, CryptoKeyType, KeyMaterial&&, bool extractable, CryptoKeyUsageBitmap);

    WEBCORE_EXPORT static ExceptionOr<CryptoKeyPair> generatePair(CryptoAlgorithmIdentifier, NamedCurve, bool extractable, CryptoKeyUsageBitmap);
    WEBCORE_EXPORT static RefPtr<CryptoKeyOKP> importRaw(CryptoAlgorithmIdentifier, NamedCurve, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap);
    static RefPtr<CryptoKeyOKP> importJwk(CryptoAlgorithmIdentifier, NamedCurve, JsonWebKey&&, bool extractable, CryptoKeyUsageBitmap);
    static RefPtr<CryptoKeyOKP> importSpki(CryptoAlgorithmIdentifier, NamedCurve, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap);
    static RefPtr<CryptoKeyOKP> importPkcs8(CryptoAlgorithmIdentifier, NamedCurve, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap);

    WEBCORE_EXPORT ExceptionOr<Vector<uint8_t>> exportRaw() const;
    ExceptionOr<JsonWebKey> exportJwk() const;
    ExceptionOr<Vector<uint8_t>> exportSpki() const;
    ExceptionOr<Vector<uint8_t>> exportPkcs8() const;

    static std::optional<NamedCurve> namedCurveFromString(const String&);
    NamedCurve namedCurve() const { return m_curve; }
    String namedCurveString() const;

    static bool isValidOKPAlgorithm(CryptoAlgorithmIdentifier);

    size_t keySizeInBits() const { return platformKey().size() * 8; }
    size_t keySizeInBytes() const { return platformKey().size(); }
    const KeyMaterial& platformKey() const { return m_data; }

private:
    CryptoKeyOKP(CryptoAlgorithmIdentifier, NamedCurve, CryptoKeyType, Vector<uint8_t>&&, bool extractable, CryptoKeyUsageBitmap);

    CryptoKeyClass keyClass() const final { return CryptoKeyClass::OKP; }
    KeyAlgorithm algorithm() const final;
    CryptoKey::Data data() const final;

    String generateJwkD() const;
    String generateJwkX() const;

    static bool supportsNamedCurve();
    static std::optional<CryptoKeyPair> platformGeneratePair(CryptoAlgorithmIdentifier, NamedCurve, bool extractable, CryptoKeyUsageBitmap);
    static bool platformCheckPairedKeys(CryptoAlgorithmIdentifier, NamedCurve, const Vector<uint8_t>&, const Vector<uint8_t>&);
    Vector<uint8_t> platformExportRaw() const;
    Vector<uint8_t> platformExportSpki() const;
    Vector<uint8_t> platformExportPkcs8() const;

    NamedCurve m_curve;
    KeyMaterial m_data;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CRYPTO_KEY(CryptoKeyOKP, CryptoKeyClass::OKP)
