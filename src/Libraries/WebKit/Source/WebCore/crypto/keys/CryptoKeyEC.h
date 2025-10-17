/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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

#if OS(DARWIN) && !PLATFORM(GTK)
#include "CommonCryptoUtilities.h"
#if HAVE(SWIFT_CPP_INTEROP)
namespace PAL {
class ECKey;
}

namespace WebCore {
using PlatformECKeyContainer = UniqueRef<PAL::ECKey>;
}
#else
namespace WebCore {
struct CCECCryptorRefDeleter {
    void operator()(CCECCryptorRef key) const { CCECCryptorRelease(key); }
};
typedef std::unique_ptr<typename std::remove_pointer<CCECCryptorRef>::type, WebCore::CCECCryptorRefDeleter> PlatformECKeyContainer;
}
#endif
#endif

#if USE(GCRYPT)
#include <pal/crypto/gcrypt/Handle.h>
namespace WebCore {
using PlatformECKeyContainer = std::unique_ptr<typename std::remove_pointer<gcry_sexp_t>::type, PAL::GCrypt::HandleDeleter<gcry_sexp_t>>;
}
#endif

#if USE(OPENSSL)
#include "crypto/openssl/OpenSSLCryptoUniquePtr.h"
namespace WebCore {
using PlatformECKeyContainer = WebCore::EvpPKeyPtr;
}
#endif

namespace WebCore {

struct JsonWebKey;

class CryptoKeyEC final : public CryptoKey {
public:
    enum class NamedCurve : uint8_t {
        P256,
        P384,
        P521,
    };

    static Ref<CryptoKeyEC> create(CryptoAlgorithmIdentifier identifier, NamedCurve curve, CryptoKeyType type, PlatformECKeyContainer&& platformKey, bool extractable, CryptoKeyUsageBitmap usages)
    {
        return adoptRef(*new CryptoKeyEC(identifier, curve, type, WTFMove(platformKey), extractable, usages));
    }
    virtual ~CryptoKeyEC();

    WEBCORE_EXPORT static ExceptionOr<CryptoKeyPair> generatePair(CryptoAlgorithmIdentifier, const String& curve, bool extractable, CryptoKeyUsageBitmap);
    WEBCORE_EXPORT static RefPtr<CryptoKeyEC> importRaw(CryptoAlgorithmIdentifier, const String& curve, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap);
    static RefPtr<CryptoKeyEC> importJwk(CryptoAlgorithmIdentifier, const String& curve, JsonWebKey&&, bool extractable, CryptoKeyUsageBitmap);
    static RefPtr<CryptoKeyEC> importSpki(CryptoAlgorithmIdentifier, const String& curve, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap);
    static RefPtr<CryptoKeyEC> importPkcs8(CryptoAlgorithmIdentifier, const String& curve, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap);

    WEBCORE_EXPORT ExceptionOr<Vector<uint8_t>> exportRaw() const;
    ExceptionOr<JsonWebKey> exportJwk() const;
    ExceptionOr<Vector<uint8_t>> exportSpki() const;
    ExceptionOr<Vector<uint8_t>> exportPkcs8() const;

    size_t keySizeInBits() const;
    size_t keySizeInBytes() const { return std::ceil(keySizeInBits() / 8.); }
    NamedCurve namedCurve() const { return m_curve; }
    String namedCurveString() const;
    const PlatformECKeyContainer& platformKey() const { return m_platformKey; }

    static bool isValidECAlgorithm(CryptoAlgorithmIdentifier);

private:
    CryptoKeyEC(CryptoAlgorithmIdentifier, NamedCurve, CryptoKeyType, PlatformECKeyContainer&&, bool extractable, CryptoKeyUsageBitmap);

    CryptoKeyClass keyClass() const final { return CryptoKeyClass::EC; }
    KeyAlgorithm algorithm() const final;
    CryptoKey::Data data() const final;

    static bool platformSupportedCurve(NamedCurve);
    static std::optional<CryptoKeyPair> platformGeneratePair(CryptoAlgorithmIdentifier, NamedCurve, bool extractable, CryptoKeyUsageBitmap);
    static RefPtr<CryptoKeyEC> platformImportRaw(CryptoAlgorithmIdentifier, NamedCurve, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap);
    static RefPtr<CryptoKeyEC> platformImportJWKPublic(CryptoAlgorithmIdentifier, NamedCurve, Vector<uint8_t>&& x, Vector<uint8_t>&& y, bool extractable, CryptoKeyUsageBitmap);
    static RefPtr<CryptoKeyEC> platformImportJWKPrivate(CryptoAlgorithmIdentifier, NamedCurve, Vector<uint8_t>&& x, Vector<uint8_t>&& y, Vector<uint8_t>&& d, bool extractable, CryptoKeyUsageBitmap);
    static RefPtr<CryptoKeyEC> platformImportSpki(CryptoAlgorithmIdentifier, NamedCurve, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap);
    static RefPtr<CryptoKeyEC> platformImportPkcs8(CryptoAlgorithmIdentifier, NamedCurve, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap);
    Vector<uint8_t> platformExportRaw() const;
    bool platformAddFieldElements(JsonWebKey&) const;
    Vector<uint8_t> platformExportSpki() const;
    Vector<uint8_t> platformExportPkcs8() const;

    PlatformECKeyContainer m_platformKey;
    NamedCurve m_curve;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CRYPTO_KEY(CryptoKeyEC, CryptoKeyClass::EC)
