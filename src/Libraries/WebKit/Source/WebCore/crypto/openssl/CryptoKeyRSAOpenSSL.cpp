/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 22, 2024.
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
#include "CryptoKeyRSA.h"

#include "CryptoAlgorithmRegistry.h"
#include "CryptoKeyPair.h"
#include "CryptoKeyRSAComponents.h"
#include "OpenSSLUtilities.h"
#include <JavaScriptCore/TypedArrayInlines.h>
#include <openssl/X509.h>
#include <openssl/evp.h>

namespace WebCore {

static size_t getRSAModulusLength(RSA* rsa)
{
    if (!rsa)
        return 0;
    return RSA_size(rsa) * 8;
}

RefPtr<CryptoKeyRSA> CryptoKeyRSA::create(CryptoAlgorithmIdentifier identifier, CryptoAlgorithmIdentifier hash, bool hasHash, const CryptoKeyRSAComponents& keyData, bool extractable, CryptoKeyUsageBitmap usages)
{
    CryptoKeyType keyType;
    switch (keyData.type()) {
    case CryptoKeyRSAComponents::Type::Public:
        keyType = CryptoKeyType::Public;
        break;
    case CryptoKeyRSAComponents::Type::Private:
        keyType = CryptoKeyType::Private;
        break;
    default:
        return nullptr;
    }

    // When creating a private key, we require the p and q prime information.
    if (keyType == CryptoKeyType::Private && !keyData.hasAdditionalPrivateKeyParameters())
        return nullptr;

    // But we don't currently support creating keys with any additional prime information.
    if (!keyData.otherPrimeInfos().isEmpty())
        return nullptr;

    // For both public and private keys, we need the public modulus and exponent.
    if (keyData.modulus().isEmpty() || keyData.exponent().isEmpty())
        return nullptr;

    // For private keys, we require the private exponent, as well as p and q prime information.
    if (keyType == CryptoKeyType::Private) {
        if (keyData.privateExponent().isEmpty() || keyData.firstPrimeInfo().primeFactor.isEmpty() || keyData.secondPrimeInfo().primeFactor.isEmpty())
            return nullptr;
    }

    auto rsa = RSAPtr(RSA_new());
    if (!rsa)
        return nullptr;

    auto n = convertToBigNumber(keyData.modulus());
    auto e = convertToBigNumber(keyData.exponent());
    if (!n || !e)
        return nullptr;

    // Calling with d null is fine as long as n and e are not null
    if (!RSA_set0_key(rsa.get(), n.get(), e.get(), nullptr))
        return nullptr;

    // Ownership transferred to OpenSSL
    n.release();
    e.release();

    if (keyType == CryptoKeyType::Private) {
        auto d = convertToBigNumber(keyData.privateExponent());
        if (!d)
            return nullptr;

        // Calling with n and e null is fine as long as they were set prior
        if (!RSA_set0_key(rsa.get(), nullptr, nullptr, d.get()))
            return nullptr;

        // Ownership transferred to OpenSSL
        d.release();

        auto p = convertToBigNumber(keyData.firstPrimeInfo().primeFactor);
        auto q = convertToBigNumber(keyData.secondPrimeInfo().primeFactor);
        if (!p || !q)
            return nullptr;

        if (!RSA_set0_factors(rsa.get(), p.get(), q.get()))
            return nullptr;

        // Ownership transferred to OpenSSL
        p.release();
        q.release();

        // We set dmp1, dmpq1, and iqmp member of the RSA struct if the keyData has corresponding data.

        // dmp1 -- d mod (p - 1)
        auto dmp1 = (!keyData.firstPrimeInfo().factorCRTExponent.isEmpty()) ? convertToBigNumber(keyData.firstPrimeInfo().factorCRTExponent) : nullptr;
        // dmq1 -- d mod (q - 1)
        auto dmq1 = (!keyData.secondPrimeInfo().factorCRTExponent.isEmpty()) ? convertToBigNumber(keyData.secondPrimeInfo().factorCRTExponent) : nullptr;
        // iqmp -- q^(-1) mod p
        auto iqmp = (!keyData.secondPrimeInfo().factorCRTCoefficient.isEmpty()) ? convertToBigNumber(keyData.secondPrimeInfo().factorCRTCoefficient) : nullptr;

        if (!RSA_set0_crt_params(rsa.get(), dmp1.get(), dmq1.get(), iqmp.get()))
            return nullptr;

        // Ownership transferred to OpenSSL
        dmp1.release();
        dmq1.release();
        iqmp.release();
    }

    auto pkey = EvpPKeyPtr(EVP_PKEY_new());
    if (!pkey)
        return nullptr;

    if (EVP_PKEY_set1_RSA(pkey.get(), rsa.get()) != 1)
        return nullptr;

    return adoptRef(new CryptoKeyRSA(identifier, hash, hasHash, keyType, WTFMove(pkey), extractable, usages));
}

CryptoKeyRSA::CryptoKeyRSA(CryptoAlgorithmIdentifier identifier, CryptoAlgorithmIdentifier hash, bool hasHash, CryptoKeyType type, PlatformRSAKeyContainer&& platformKey, bool extractable, CryptoKeyUsageBitmap usages)
    : CryptoKey(identifier, type, extractable, usages)
    , m_platformKey(WTFMove(platformKey))
    , m_restrictedToSpecificHash(hasHash)
    , m_hash(hash)
{
}

bool CryptoKeyRSA::isRestrictedToHash(CryptoAlgorithmIdentifier& identifier) const
{
    if (!m_restrictedToSpecificHash)
        return false;

    identifier = m_hash;
    return true;
}

size_t CryptoKeyRSA::keySizeInBits() const
{
    RSA* rsa = EVP_PKEY_get0_RSA(m_platformKey.get());
    if (!rsa)
        return 0;

    return getRSAModulusLength(rsa);
}

// Convert the exponent vector to a 32-bit value, if possible.
static std::optional<uint32_t> exponentVectorToUInt32(const Vector<uint8_t>& exponent)
{
    if (exponent.size() > 4) {
        if (std::any_of(exponent.begin(), exponent.end() - 4, [](uint8_t element) { return !!element; }))
            return std::nullopt;
    }

    uint32_t result = 0;
    for (size_t size = exponent.size(), i = std::min<size_t>(4, size); i > 0; --i) {
        result <<= 8;
        result += exponent[size - i];
    }

    return result;
}

void CryptoKeyRSA::generatePair(CryptoAlgorithmIdentifier algorithm, CryptoAlgorithmIdentifier hash, bool hasHash, unsigned modulusLength, const Vector<uint8_t>& publicExponent, bool extractable, CryptoKeyUsageBitmap usages, KeyPairCallback&& callback, VoidCallback&& failureCallback, ScriptExecutionContext*)
{
    // OpenSSL doesn't report an error if the exponent is smaller than three or even.
    auto e = exponentVectorToUInt32(publicExponent);
    if (!e || *e < 3 || !(*e & 0x1)) {
        failureCallback();
        return;
    }

    auto exponent = convertToBigNumber(publicExponent);
    auto privateRSA = RSAPtr(RSA_new());
    if (!exponent || RSA_generate_key_ex(privateRSA.get(), modulusLength, exponent.get(), nullptr) <= 0) {
        failureCallback();
        return;
    }

    auto publicRSA = RSAPtr(RSAPublicKey_dup(privateRSA.get()));
    if (!publicRSA) {
        failureCallback();
        return;
    }

    auto privatePKey = EvpPKeyPtr(EVP_PKEY_new());
    if (EVP_PKEY_set1_RSA(privatePKey.get(), privateRSA.get()) <= 0) {
        failureCallback();
        return;
    }

    auto publicPKey = EvpPKeyPtr(EVP_PKEY_new());
    if (EVP_PKEY_set1_RSA(publicPKey.get(), publicRSA.get()) <= 0) {
        failureCallback();
        return;
    }

    auto publicKey = CryptoKeyRSA::create(algorithm, hash, hasHash, CryptoKeyType::Public, WTFMove(publicPKey), true, usages);
    auto privateKey = CryptoKeyRSA::create(algorithm, hash, hasHash, CryptoKeyType::Private, WTFMove(privatePKey), extractable, usages);
    callback(CryptoKeyPair { WTFMove(publicKey), WTFMove(privateKey) });
}

RefPtr<CryptoKeyRSA> CryptoKeyRSA::importSpki(CryptoAlgorithmIdentifier identifier, std::optional<CryptoAlgorithmIdentifier> hash, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap usages)
{
    // We need a local pointer variable to pass to d2i (DER to internal) functions().
    const uint8_t* ptr = keyData.data();

    // We use d2i_PUBKEY() to import a public key.
    auto pkey = EvpPKeyPtr(d2i_PUBKEY(nullptr, &ptr, keyData.size()));
    if (!pkey || EVP_PKEY_id(pkey.get()) != EVP_PKEY_RSA)
        return nullptr;

    return adoptRef(new CryptoKeyRSA(identifier, hash.value_or(CryptoAlgorithmIdentifier::SHA_1), !!hash, CryptoKeyType::Public, WTFMove(pkey), extractable, usages));
}

RefPtr<CryptoKeyRSA> CryptoKeyRSA::importPkcs8(CryptoAlgorithmIdentifier identifier, std::optional<CryptoAlgorithmIdentifier> hash, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap usages)
{
    // We need a local pointer variable to pass to d2i (DER to internal) functions().
    const uint8_t* ptr = keyData.data();

    // We use d2i_PKCS8_PRIV_KEY_INFO() to import a private key.
    auto p8inf = PKCS8PrivKeyInfoPtr(d2i_PKCS8_PRIV_KEY_INFO(nullptr, &ptr, keyData.size()));
    if (!p8inf)
        return nullptr;

    auto pkey = EvpPKeyPtr(EVP_PKCS82PKEY(p8inf.get()));
    if (!pkey || EVP_PKEY_id(pkey.get()) != EVP_PKEY_RSA)
        return nullptr;

    return adoptRef(new CryptoKeyRSA(identifier, hash.value_or(CryptoAlgorithmIdentifier::SHA_1), !!hash, CryptoKeyType::Private, WTFMove(pkey), extractable, usages));
}

ExceptionOr<Vector<uint8_t>> CryptoKeyRSA::exportSpki() const
{
    if (type() != CryptoKeyType::Public)
        return Exception { ExceptionCode::InvalidAccessError };

    int len = i2d_PUBKEY(platformKey(), nullptr);
    if (len < 0)
        return Exception { ExceptionCode::OperationError };

    Vector<uint8_t> keyData(len);
    auto ptr = keyData.data();
    if (i2d_PUBKEY(platformKey(), &ptr) < 0)
        return Exception { ExceptionCode::OperationError };

    return keyData;
}

ExceptionOr<Vector<uint8_t>> CryptoKeyRSA::exportPkcs8() const
{
    if (type() != CryptoKeyType::Private)
        return Exception { ExceptionCode::InvalidAccessError };

    auto p8inf = PKCS8PrivKeyInfoPtr(EVP_PKEY2PKCS8(platformKey()));
    if (!p8inf)
        return Exception { ExceptionCode::OperationError };

    int len = i2d_PKCS8_PRIV_KEY_INFO(p8inf.get(), nullptr);
    if (len < 0)
        return Exception { ExceptionCode::OperationError };

    Vector<uint8_t> keyData(len);
    auto ptr = keyData.data();
    if (i2d_PKCS8_PRIV_KEY_INFO(p8inf.get(), &ptr) < 0)
        return Exception { ExceptionCode::OperationError };

    return keyData;
}

auto CryptoKeyRSA::algorithm() const -> KeyAlgorithm
{
    RSA* rsa = EVP_PKEY_get0_RSA(platformKey());

    auto modulusLength = getRSAModulusLength(rsa);
    Vector<uint8_t> publicExponent;

    if (rsa) {
        const BIGNUM* e;
        RSA_get0_key(rsa, nullptr, &e, nullptr);
        publicExponent = convertToBytes(e);
    }

    if (m_restrictedToSpecificHash) {
        CryptoRsaHashedKeyAlgorithm result;
        result.name = CryptoAlgorithmRegistry::singleton().name(algorithmIdentifier());
        result.modulusLength = modulusLength;
        result.publicExponent = Uint8Array::tryCreate(publicExponent.data(), publicExponent.size());
        result.hash.name = CryptoAlgorithmRegistry::singleton().name(m_hash);
        return result;
    }

    CryptoRsaKeyAlgorithm result;
    result.name = CryptoAlgorithmRegistry::singleton().name(algorithmIdentifier());
    result.modulusLength = modulusLength;
    result.publicExponent = Uint8Array::tryCreate(publicExponent.data(), publicExponent.size());
    return result;
}

std::unique_ptr<CryptoKeyRSAComponents> CryptoKeyRSA::exportData() const
{
    RSA* rsa = EVP_PKEY_get0_RSA(platformKey());
    if (!rsa)
        return nullptr;

    const BIGNUM* n;
    const BIGNUM* e;
    const BIGNUM* d;
    RSA_get0_key(rsa, &n, &e, &d);

    switch (type()) {
    case CryptoKeyType::Public:
        // We need the public modulus and exponent for the public key.
        if (!n || !e)
            return nullptr;
        return CryptoKeyRSAComponents::createPublic(convertToBytes(n), convertToBytes(e));
    case CryptoKeyType::Private: {
        // We need the public modulus, exponent, and private exponent, as well as p and q prime information.
        const BIGNUM* p;
        const BIGNUM* q;
        RSA_get0_factors(rsa, &p, &q);

        if (!n || !e || !d || !p || !q)
            return nullptr;

        CryptoKeyRSAComponents::PrimeInfo firstPrimeInfo;
        firstPrimeInfo.primeFactor = convertToBytes(p);

        CryptoKeyRSAComponents::PrimeInfo secondPrimeInfo;
        secondPrimeInfo.primeFactor = convertToBytes(q);

        auto context = BNCtxPtr(BN_CTX_new());

        const BIGNUM* dmp1;
        const BIGNUM* dmq1;
        const BIGNUM* iqmp;
        RSA_get0_crt_params(rsa, &dmp1, &dmq1, &iqmp);

        // dmp1 -- d mod (p - 1)
        if (dmp1)
            firstPrimeInfo.factorCRTExponent = convertToBytes(dmp1);
        else {
            auto dmp1New = BIGNUMPtr(BN_new());
            auto pm1 = BIGNUMPtr(BN_dup(p));
            if (BN_sub_word(pm1.get(), 1) == 1 && BN_mod(dmp1New.get(), d, pm1.get(), context.get()) == 1)
                firstPrimeInfo.factorCRTExponent = convertToBytes(dmp1New.get());
        }

        // dmq1 -- d mod (q - 1)
        if (dmq1)
            secondPrimeInfo.factorCRTExponent = convertToBytes(dmq1);
        else {
            auto dmq1New = BIGNUMPtr(BN_new());
            auto qm1 = BIGNUMPtr(BN_dup(q));
            if (BN_sub_word(qm1.get(), 1) == 1 && BN_mod(dmq1New.get(), d, qm1.get(), context.get()) == 1)
                secondPrimeInfo.factorCRTExponent = convertToBytes(dmq1New.get());
        }

        // iqmp -- q^(-1) mod p
        if (iqmp)
            secondPrimeInfo.factorCRTCoefficient = convertToBytes(iqmp);
        else {
            auto iqmpNew = BIGNUMPtr(BN_mod_inverse(nullptr, q, p, context.get()));
            if (iqmpNew)
                secondPrimeInfo.factorCRTCoefficient = convertToBytes(iqmpNew.get());
        }

        return CryptoKeyRSAComponents::createPrivateWithAdditionalData(
            convertToBytes(n), convertToBytes(e), convertToBytes(d),
            WTFMove(firstPrimeInfo), WTFMove(secondPrimeInfo), Vector<CryptoKeyRSAComponents::PrimeInfo> { });
    }
    default:
        ASSERT_NOT_REACHED();
        return nullptr;
    }
}

} // namespace WebCore
