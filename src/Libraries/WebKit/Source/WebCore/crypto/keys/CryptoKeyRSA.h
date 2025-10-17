/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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
#include "ExceptionOr.h"
#include <wtf/Function.h>

#if OS(DARWIN) && !PLATFORM(GTK)
#include "CommonCryptoUtilities.h"

typedef CCRSACryptorRef PlatformRSAKey;
namespace WebCore {
struct CCRSACryptorRefDeleter {
    void operator()(CCRSACryptorRef key) const { CCRSACryptorRelease(key); }
};
}
typedef std::unique_ptr<typename std::remove_pointer<CCRSACryptorRef>::type, WebCore::CCRSACryptorRefDeleter> PlatformRSAKeyContainer;
#endif

#if USE(GCRYPT)
#include <pal/crypto/gcrypt/Handle.h>

typedef gcry_sexp_t PlatformRSAKey;
typedef std::unique_ptr<typename std::remove_pointer<gcry_sexp_t>::type, PAL::GCrypt::HandleDeleter<gcry_sexp_t>> PlatformRSAKeyContainer;
#endif

#if USE(OPENSSL)
#include "crypto/openssl/OpenSSLCryptoUniquePtr.h"
typedef EVP_PKEY* PlatformRSAKey;
typedef WebCore::EvpPKeyPtr PlatformRSAKeyContainer;
#endif

namespace WebCore {

class CryptoKeyRSAComponents;
class PromiseWrapper;
class ScriptExecutionContext;

struct CryptoKeyPair;
struct JsonWebKey;

class CryptoKeyRSA final : public CryptoKey {
public:
    static Ref<CryptoKeyRSA> create(CryptoAlgorithmIdentifier identifier, CryptoAlgorithmIdentifier hash, bool hasHash, CryptoKeyType type, PlatformRSAKeyContainer&& platformKey, bool extractable, CryptoKeyUsageBitmap usage)
    {
        return adoptRef(*new CryptoKeyRSA(identifier, hash, hasHash, type, WTFMove(platformKey), extractable, usage));
    }
    static RefPtr<CryptoKeyRSA> create(CryptoAlgorithmIdentifier, CryptoAlgorithmIdentifier hash, bool hasHash, const CryptoKeyRSAComponents&, bool extractable, CryptoKeyUsageBitmap);
    virtual ~CryptoKeyRSA() = default;

    bool isRestrictedToHash(CryptoAlgorithmIdentifier&) const;

    size_t keySizeInBits() const;

    using KeyPairCallback = Function<void(CryptoKeyPair&&)>;
    using VoidCallback = Function<void()>;
    static void generatePair(CryptoAlgorithmIdentifier, CryptoAlgorithmIdentifier hash, bool hasHash, unsigned modulusLength, const Vector<uint8_t>& publicExponent, bool extractable, CryptoKeyUsageBitmap, KeyPairCallback&&, VoidCallback&& failureCallback, ScriptExecutionContext*);
    static RefPtr<CryptoKeyRSA> importJwk(CryptoAlgorithmIdentifier, std::optional<CryptoAlgorithmIdentifier> hash, JsonWebKey&&, bool extractable, CryptoKeyUsageBitmap);
    static RefPtr<CryptoKeyRSA> importSpki(CryptoAlgorithmIdentifier, std::optional<CryptoAlgorithmIdentifier> hash, Vector<uint8_t>&&, bool extractable, CryptoKeyUsageBitmap);
    static RefPtr<CryptoKeyRSA> importPkcs8(CryptoAlgorithmIdentifier, std::optional<CryptoAlgorithmIdentifier> hash, Vector<uint8_t>&&, bool extractable, CryptoKeyUsageBitmap);

    PlatformRSAKey platformKey() const { return m_platformKey.get(); }
    JsonWebKey exportJwk() const;
    ExceptionOr<Vector<uint8_t>> exportSpki() const;
    ExceptionOr<Vector<uint8_t>> exportPkcs8() const;

    std::unique_ptr<CryptoKeyRSAComponents> exportData() const;

    CryptoAlgorithmIdentifier hashAlgorithmIdentifier() const { return m_hash; }

private:
    CryptoKeyRSA(CryptoAlgorithmIdentifier, CryptoAlgorithmIdentifier hash, bool hasHash, CryptoKeyType, PlatformRSAKeyContainer&&, bool extractable, CryptoKeyUsageBitmap);

    CryptoKeyClass keyClass() const final { return CryptoKeyClass::RSA; }
    KeyAlgorithm algorithm() const final;
    CryptoKey::Data data() const final;

    PlatformRSAKeyContainer m_platformKey;

    bool m_restrictedToSpecificHash;
    CryptoAlgorithmIdentifier m_hash;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CRYPTO_KEY(CryptoKeyRSA, CryptoKeyClass::RSA)
